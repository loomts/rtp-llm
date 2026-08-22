#pragma once

#include <cstring>
#include <string>
#include <signal.h>

namespace rtp_llm {

bool installSighandler();

void printSignalStackTrace(int signum, siginfo_t* siginfo, void* ucontext);

// Logs the active exception's type and what() before aborting. Installed by
// installSighandler(), because the SIGABRT path alone cannot recover the
// message: an exception that escapes a thread reaches std::terminate, and the
// default handler aborts with a stack made only of libstdc++ frames.
void terminateHandler();

};  // namespace rtp_llm
