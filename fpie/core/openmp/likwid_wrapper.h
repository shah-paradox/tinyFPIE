#pragma once

#ifdef USE_LIKWID
#include <likwid-marker.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_START(name)
#define LIKWID_MARKER_STOP(name)
#define LIKWID_MARKER_WRITE_FILE(path)
#define LIKWID_MARKER_CLOSE
#endif
