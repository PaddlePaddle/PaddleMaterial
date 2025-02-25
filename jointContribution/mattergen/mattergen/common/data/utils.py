import os
import signal


def term_mp(sig_num, frame):
    """kill all child processes"""
    pid = os.getpid()
    pgid = os.getpgid(os.getpid())
    print("main proc {} exit, kill process group " "{}".format(pid, pgid))
    os.killpg(pgid, signal.SIGKILL)


def set_signal_handlers():
    pid = os.getpid()
    try:
        pgid = os.getpgid(pid)
    except AttributeError:
        # In case `os.getpgid` is not available, no signal handler will be set,
        # because we cannot do safe cleanup.
        pass
    else:
        # XXX: `term_mp` kills all processes in the process group, which in
        # some cases includes the parent process of current process and may
        # cause unexpected results. To solve this problem, we set signal
        # handlers only when current process is the group leader. In the
        # future, it would be better to consider killing only descendants of
        # the current process.
        if pid == pgid:
            # support exit using ctrl+c
            signal.signal(signal.SIGINT, term_mp)
            signal.signal(signal.SIGTERM, term_mp)
