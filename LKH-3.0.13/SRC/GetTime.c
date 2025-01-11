#ifdef _WIN32
#include <windows.h>
#include <time.h>
#include <winsock.h>  // timeval이 이미 정의되어 있음

struct rusage {
    struct timeval ru_utime;  // 이미 정의된 timeval 구조체 사용
    struct timeval ru_stime;
};

int getrusage(int who, struct rusage *usage) {
    if (!usage) return -1;

    FILETIME createTime, exitTime, kernelTime, userTime;
    GetProcessTimes(GetCurrentProcess(), &createTime, &exitTime, &kernelTime, &userTime);

    ULARGE_INTEGER ktime, utime;
    ktime.LowPart = kernelTime.dwLowDateTime;
    ktime.HighPart = kernelTime.dwHighDateTime;
    utime.LowPart = userTime.dwLowDateTime;
    utime.HighPart = userTime.dwHighDateTime;

    usage->ru_utime.tv_sec = utime.QuadPart / 10000000;
    usage->ru_utime.tv_usec = (utime.QuadPart % 10000000) / 10;
    usage->ru_stime.tv_sec = ktime.QuadPart / 10000000;
    usage->ru_stime.tv_usec = (ktime.QuadPart % 10000000) / 10;

    return 0;
}

double GetTime() {
    struct rusage ru;
    getrusage(0, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1000000.0;
}

#else
#include <sys/resource.h>
#include <sys/time.h>

double GetTime() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1000000.0;
}
#endif
