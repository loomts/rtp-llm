#include <csignal>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <typeinfo>
#include <unistd.h>
#include <iostream>
#include <execinfo.h>
#include <dlfcn.h>
#include <sstream>

#include "rtp_llm/cpp/utils/SignalUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/StackTrace.h"

namespace rtp_llm {

void printSignalStackTrace(int signum, siginfo_t* siginfo, void* ucontext) {
    std::stringstream stack_ss;
    time_t            current_time = time(nullptr);
    stack_ss << std::endl
             << "*** Aborted at " << current_time << " (unix time) try \"date -d @" << current_time
             << "\" if you are using GNU date***" << std::endl;

    switch (signum) {
        case SIGSEGV:
            stack_ss << "*** SIGSEGV (@0x" << std::hex << reinterpret_cast<uintptr_t>(siginfo->si_addr) << std::dec
                     << ") received by PID " << getpid() << " (TID " << gettid() << "); stack trace: ***" << std::endl;
            break;
        case SIGFPE:
            stack_ss << "*** SIGFPE (@0x" << std::hex << reinterpret_cast<uintptr_t>(siginfo->si_addr) << std::dec
                     << ") received by PID " << getpid() << " (TID " << gettid() << "); stack trace: ***" << std::endl;
            break;
        case SIGILL:
            stack_ss << "*** SIGILL (@0x" << std::hex << reinterpret_cast<uintptr_t>(siginfo->si_addr) << std::dec
                     << ") received by PID " << getpid() << " (TID " << gettid() << "); stack trace: ***" << std::endl;
            break;
        case SIGABRT:
            stack_ss << "*** SIGABRT (@0x" << std::hex << reinterpret_cast<uintptr_t>(siginfo->si_addr) << std::dec
                     << ") received by PID " << getpid() << " (TID " << gettid() << "); stack trace: ***" << std::endl;
            break;
        case SIGBUS:
            stack_ss << "*** SIGBUS (@0x" << std::hex << reinterpret_cast<uintptr_t>(siginfo->si_addr) << std::dec
                     << ") received by PID " << getpid() << " (TID " << gettid() << "); stack trace: ***" << std::endl;
            break;
        default:
            stack_ss << "*** Unknown signal (" << signum << ") received by PID " << getpid() << " (TID " << gettid()
                     << "); stack trace: ***" << std::endl;
            break;
    }
    RTP_LLM_STACKTRACE_LOG_INFO("%s", stack_ss.str().c_str());

    rtp_llm::printStackTrace();
}

void flushLog() {
    Logger::getEngineLogger().flush();
    Logger::getStackTraceLogger().flush();
    Logger::getAccessLogger().flush();
}

void getSighandler(int signum, siginfo_t* siginfo, void* ucontext) {
    printSignalStackTrace(signum, siginfo, ucontext);
    flushLog();
    signal(signum, SIG_DFL);
    kill(getpid(), signum);
}

void terminateHandler() {
    // An exception that escapes a thread entry function -- or is thrown while
    // another is propagating, e.g. from a tensor destructor that hits a device
    // error -- reaches std::terminate, whose default handler aborts. The SIGABRT
    // handler above then prints a stack trace made entirely of libstdc++ frames
    // and the message is lost, which reads as an unattributable "SIGABRT in an
    // unknown thread". Recover type and what() first, then abort as before.
    std::string detail = "no active exception";
    if (std::current_exception()) {
        try {
            std::rethrow_exception(std::current_exception());
        } catch (const std::exception& e) {
            detail = std::string(typeid(e).name()) + ": " + e.what();
        } catch (...) {
            detail = "non-std exception";
        }
    }
    RTP_LLM_LOG_ERROR("std::terminate called in PID %d (TID %d): %s", (int)getpid(), (int)gettid(), detail.c_str());
    std::stringstream stack_ss;
    stack_ss << std::endl
             << "*** std::terminate in PID " << getpid() << " (TID " << gettid() << "): " << detail
             << "; stack trace: ***" << std::endl;
    RTP_LLM_STACKTRACE_LOG_INFO("%s", stack_ss.str().c_str());
    printStackTrace();
    flushLog();
    std::abort();
}

bool installSighandler() {
    std::set_terminate(terminateHandler);

    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_sigaction = getSighandler;
    action.sa_flags     = SA_SIGINFO;
    sigfillset(&action.sa_mask);

    if (sigaction(SIGSEGV, &action, nullptr) != 0)
        return false;
    if (sigaction(SIGFPE, &action, nullptr) != 0)
        return false;
    if (sigaction(SIGILL, &action, nullptr) != 0)
        return false;
    if (sigaction(SIGABRT, &action, nullptr) != 0)
        return false;
    if (sigaction(SIGBUS, &action, nullptr) != 0)
        return false;

    return true;
}

};  // namespace rtp_llm