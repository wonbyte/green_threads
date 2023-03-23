// When Rust compiles a function, it adds a small prologue and epilogue to each function
// and this causes some issues for us when we switch contexts since we end up with a misaligned stack.
// Marking the a function as #[naked]removes the prologue and epilogue.
#![feature(naked_functions)]
use std::arch::asm;

// 2MB
const DEFAULT_STACK_SIZE: usize = 1024 * 1024 * 2;
// 4 threads is enough for our example.
const MAX_THREADS: usize = 4;
// Pointer to our runtime, we're only setting this variable on initialization.
static mut RUNTIME: usize = 0;

/// States threads can assume.
#[derive(Eq, PartialEq, Debug)]
enum State {
    // Thread is available and ready to be assigned a task if needed.
    Available,
    // Thread is running.
    Running,
    // Thread is ready to move forward and resume execution.
    Ready,
}

/// Data for a thread.
struct Thread {
    // Each thread has an id so we can separate them from each other.
    id: usize,
    // Each thread has a stack of 2MB.
    stack: Vec<u8>,
    // Each thread has a context representing the data our CPU needs to resume where it left off on a stack.
    ctx: ThreadContext,
    // Each thread has a state.
    state: State,
}

impl Thread {
    /// A new thread starts in the `Available` state indicating it is ready to be assigned a task.
    fn new(id: usize) -> Self {
        Self {
            id,
            // TODO: use `into_boxed_slice` to avoid any potential reallocation?
            stack: vec![0_u8; DEFAULT_STACK_SIZE],
            ctx: ThreadContext::default(),
            state: State::Available,
        }
    }
}

#[derive(Debug, Default)]
#[repr(C)]
/// Registers the CPU needs to resume execution on a stack.
struct ThreadContext {
    // stack pointer
    rsp: u64,
    // frame pointer
    rbp: u64,
    // callee saved ...
    rbx: u64,
    r15: u64,
    r14: u64,
    r13: u64,
    r12: u64,
}

/// A very small, simple runtime to schedule and switch between threads.
pub struct Runtime {
    threads: Vec<Thread>,
    // Which thread is currently running.
    current: usize,
}

impl Runtime {
    /// When we instantiate our `Runtime` we set up a base thread. This thread will
    /// be set to the `Running` state and will make sure we keep the runtime running
    /// until all tasks are finished.
    fn new() -> Self {
        // This will be our base thread, which will be initialized in the `Running` state.
        let base_thread = Thread {
            id: 0,
            stack: vec![0_u8; DEFAULT_STACK_SIZE],
            ctx: ThreadContext::default(),
            state: State::Running,
        };

        let mut threads = vec![base_thread];

        // Instantiate the rest of the threads.
        let mut available_threads: Vec<Thread> = (1..MAX_THREADS).map(|i| Thread::new(i)).collect();
        threads.append(&mut available_threads);

        Self {
            threads,
            current: 0,
        }
    }

    /// This is cheating a bit, but we need a pointer to our `Runtime` stored
    /// so we can call yield on it even if we don't have a reference to it.
    /// We know that our `Runtime` will be alive as long as there is any thread to
    /// yield so as long as we don't abuse this it's safe to do.
    pub fn init(&self) {
        unsafe {
            let r_ptr: *const Runtime = self;
            RUNTIME = r_ptr as usize;
        }
    }

    /// Return function that we call when the thread is finished.
    ///
    /// The user of our threads does not call this, we set up our stack so this
    /// is called when the task is done.
    fn t_return(&mut self) {
        // If the calling thread is the base_thread we don't do anything our `Runtime`
        // will call yield for us on the base thread.
        //
        // If it's called from a spawned thread we know it's finished since all
        // threads have a guard function on top of their stack and the only place
        // this function is called is on our guard function.
        if self.current != 0 {
            // Set state to `Available` letting the runtime know it's ready to be assigned a new task.
            self.threads[self.current].state = State::Available;
            // Immediately call `t_yield` which will schedule a new thread to be run.
            self.t_yield();
        }
    }

    /// Schedule a new thread to be run.
    #[inline(never)]
    fn t_yield(&mut self) -> bool {
        let mut pos = self.current;
        // Go through all the threads and see if anyone is in the `Ready` state
        // which indicates it has a task it is ready to make progress on.
        while self.threads[pos].state != State::Ready {
            pos += 1;
            if pos == self.threads.len() {
                pos = 0;
            }

            if pos == self.current {
                return false;
            }
        }

        // If we find a thread that's ready to be run we change the state of the
        // current thread from Running to Ready.
        if self.threads[self.current].state != State::Available {
            self.threads[self.current].state = State::Ready;
        }

        self.threads[pos].state = State::Running;
        let old_pos = self.current;
        self.current = pos;

        // We call switch which will save the current context (the old context) and
        // load the new context into the CPU. The new context is either a new task,
        // or all the information the CPU needs to resume work on an existing task.
        unsafe {
            let old: *mut ThreadContext = &mut self.threads[old_pos].ctx;
            let new: *mut ThreadContext = &mut self.threads[pos].ctx;
            // On Linux, the ABI states that the "rdi" register holds the first argument
            // to a function, and "rsi" holds the second argument.
            //
            // The compiler will push the values of these registers on to the stack
            // before calling switch and pop them back in to the same registers once the function returns.
            //
            // Pass in the address to our "old" and "new" `ThreadContext` using assembly.
            asm!("call switch", in("rdi") old, in("rsi") new, clobber_abi("C"));
        }

        // A way for us to prevent the compiler from optimizing our code away.
        // The code never reaches this point.
        // TODO: see https://doc.rust-lang.org/std/hint/fn.black_box.html
        self.threads.len() > 0
    }

    /// Start running our `Runtime`. It will continually call `t_yield()` until
    /// it returns false which means that there is no more work to do and we can exit the process.
    pub fn run(&mut self) -> ! {
        while self.t_yield() {}
        std::process::exit(0);
    }

    /// Set up our stack to adhere to the one specified in the X86-psABI.
    pub fn spawn(&mut self, f: fn()) {
        // When we spawn a new thread we first check if there are any available threads (threads in Available state).
        // If we run out of threads we panic.
        let available = self
            .threads
            .iter_mut()
            .find(|t| t.state == State::Available)
            .expect("no available thread");

        // The stack length.
        let size = available.stack.len();
        unsafe {
            // A pointer to our u8 byte array.
            let s_ptr = available.stack.as_mut_ptr().offset(size as isize);
            // Make sure that the memory segment we'll use is 16-byte aligned.
            let s_ptr = (s_ptr as usize & !15) as *mut u8;
            // write the address to our guard function that will be called when
            // the task we provide finishes and the function returns.
            std::ptr::write(s_ptr.offset(-16) as *mut u64, guard as u64);
            // Write the address to a `skip` function which is there just to handle
            // the gap when we return from `f` so that guard will get called on a 16 byte boundary.
            std::ptr::write(s_ptr.offset(-24) as *mut u64, skip as u64);
            std::ptr::write(s_ptr.offset(-32) as *mut u64, f as u64);
            // We set the value of `rsp` which is the stack pointer to the address
            // of our provided function, so we start executing that first when we are scheduled to run.
            available.ctx.rsp = s_ptr.offset(-32) as u64;
        }
        // Set the state as `Ready` which means we have work to do and that we are ready to do it.
        // It's up to our "scheduler" to actually start up this thread.
        available.state = State::Ready;
    }
}

/// The function means that the function we passed in has returned and that means
/// our thread is finished running its task.
fn guard() {
    unsafe {
        let rt_ptr = RUNTIME as *mut Runtime;
        // Marks our thread as `Available` (if it's not our base thread) and yields
        // so we can resume work on a different thread.
        (*rt_ptr).t_return();
    };
}

#[naked]
/// This will just pop off the next value from the stack and jump to whatever instructions
/// that address points to. In our case this is the `guard` function.
unsafe extern "C" fn skip() {
    // noreturn prevents it putting a 'ret' at the end.
    asm!("ret", options(noreturn))
}

/// Call yield from an arbitrary place in our code.
pub fn yield_thread() {
    unsafe {
        let rt_ptr = RUNTIME as *mut Runtime;
        (*rt_ptr).t_yield();
    }
}

#[naked]
// Tagged with `#[no_mangle]` so we can call it by name.
#[no_mangle]
/// Our stack switch in inline Assembly. We first read out the values of all the
/// registers we need and then sets all the register values to the register values
/// we saved when we suspended execution on the "new" thread.
///
/// We only offset the pointer in 8 byte steps which is the same size as the u64 fields
/// on our `ThreadContext` struct.
unsafe extern "C" fn switch() {
    asm!(
        "mov [rdi + 0x00], rsp", // 0
        "mov [rdi + 0x08], r15", // 8
        "mov [rdi + 0x10], r14", // 16
        "mov [rdi + 0x18], r13", // 24
        "mov [rdi + 0x20], r12",
        "mov [rdi + 0x28], rbx",
        "mov [rdi + 0x30], rbp",
        "mov rsp, [rsi + 0x00]",
        "mov r15, [rsi + 0x08]",
        "mov r14, [rsi + 0x10]",
        "mov r13, [rsi + 0x18]",
        "mov r12, [rsi + 0x20]",
        "mov rbx, [rsi + 0x28]",
        "mov rbp, [rsi + 0x30]",
        "ret",
        // noreturn prevents it putting a 'ret' at the end.
        options(noreturn)
    );
}

// Initialize our runtime and spawn two threads one that counts to 10 and yields
// between each count, and one that counts to 15.
fn main() {
    let mut runtime = Runtime::new();
    runtime.init();
    runtime.spawn(|| {
        println!("THREAD 1 STARTING");
        let id = 1;
        for i in 0..10 {
            println!("thread: {} counter: {}", id, i);
            yield_thread();
        }
        println!("THREAD 1 FINISHED");
    });
    runtime.spawn(|| {
        println!("THREAD 2 STARTING");
        let id = 2;
        for i in 0..15 {
            println!("thread: {} counter: {}", id, i);
            yield_thread();
        }
        println!("THREAD 2 FINISHED");
    });
    runtime.run();
}
