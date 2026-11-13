// Deterministic Xwayland damage fixture. Rendering uses X11 rectangles, not swaps.
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/cursorfont.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <poll.h>
#include <unistd.h>
#include <signal.h>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

static volatile sig_atomic_t running = 1;
static void stop(int) { running = 0; }
static double number(const char *s) {
    char *end; double v = strtod(s, &end);
    if (!*s || *end || !std::isfinite(v)) throw std::runtime_error("invalid number");
    return v;
}

int main(int argc, char **argv) {
    Display *display = nullptr;
    int server = -1;
    std::string control, ready;
    try {
        double percent = 50, fps = 60, duration = 30;
        bool controlled = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help") {
                puts("Usage: gamescope-damage [--damage-percent 0..100] [--fps 60] [--duration 30] [--control SOCKET --ready FILE]\n"
                     "Control: state PERCENT NIGHT(0|1), status, quit. Mouse input is supplied by gamescope-test-mouse.");
                return 0;
            }
            if (i + 1 == argc) throw std::runtime_error("missing option value");
            const char *v = argv[++i];
            if (arg == "--damage-percent") percent = number(v);
            else if (arg == "--fps") fps = number(v);
            else if (arg == "--duration") duration = number(v);
            else if (arg == "--control") { control = v; controlled = true; }
            else if (arg == "--ready") ready = v;
            else throw std::runtime_error("unknown option: " + arg);
        }
        if (percent < 0 || percent > 100 || fps <= 0 || fps > 1000 || duration <= 0)
            throw std::runtime_error("damage must be 0..100, FPS 0..1000, duration positive");
        signal(SIGINT, stop); signal(SIGTERM, stop); signal(SIGHUP, stop); signal(SIGPIPE, SIG_IGN);
        display = XOpenDisplay(nullptr);
        if (!display) throw std::runtime_error("cannot open DISPLAY");
        int screen = DefaultScreen(display), width = DisplayWidth(display, screen), height = DisplayHeight(display, screen);
        Window root = RootWindow(display, screen);
        Window window = XCreateSimpleWindow(display, root, 0, 0, width, height, 0, 0, 0x102030);
        XStoreName(display, window, "Gamescope damage benchmark");
        unsigned long pid = getpid();
        XChangeProperty(display, window, XInternAtom(display, "_NET_WM_PID", False), XA_CARDINAL, 32,
                        PropModeReplace, reinterpret_cast<unsigned char *>(&pid), 1);
        XSelectInput(display, window, ExposureMask | StructureNotifyMask);
        Cursor cursor = XCreateFontCursor(display, XC_left_ptr);
        XDefineCursor(display, window, cursor);
        XMapWindow(display, window);
        GC gc = XCreateGC(display, window, 0, nullptr);
        XSync(display, False);
        if (controlled) {
            if (control.size() >= sizeof(sockaddr_un{}.sun_path)) throw std::runtime_error("socket path too long");
            server = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
            if (server < 0) throw std::runtime_error("socket failed");
            sockaddr_un address{}; address.sun_family = AF_UNIX;
            memcpy(address.sun_path, control.c_str(), control.size() + 1);
            if (bind(server, reinterpret_cast<sockaddr *>(&address), sizeof(address)) || listen(server, 4))
                throw std::runtime_error("bind/listen failed");
        }
        int objectWidth = 1, objectHeight = 1, x = 0, y = 0, dx = 2, dy = 2;
        uint64_t frames = 0;
        double actual = 0;
        bool night = false;
        auto configure = [&] {
            const double area = width * double(height) * percent / 100;
            if (percent == 100) { objectWidth = width; objectHeight = height; }
            else {
                objectWidth = std::clamp(int(std::sqrt(area * width / height)) - 2, 1, width - 2);
                objectHeight = std::clamp(int(std::lround(area / (objectWidth + 2))) - 2, 1, height - 2);
            }
            actual = percent == 0 ? 0 : percent == 100 ? 100 :
                100.0 * (objectWidth + 2) * (objectHeight + 2) / (width * double(height));
            x = y = 0; dx = dy = 2;
            XSetForeground(display, gc, 0x102030); XFillRectangle(display, window, gc, 0, 0, width, height);
            float values[] = {night ? .75f : 0.f, .06944445f, .95f};
            unsigned long property[3];
            for (int i = 0; i < 3; ++i) { uint32_t bits; memcpy(&bits, &values[i], 4); property[i] = bits; }
            XChangeProperty(display, root, XInternAtom(display, "GAMESCOPE_COLOR_NIGHT_MODE", False), XA_CARDINAL,
                            32, PropModeReplace, reinterpret_cast<unsigned char *>(property), 3);
            XSync(display, False);
        };
        auto status = [&] {
            return std::string("{\"pid\":") + std::to_string(getpid()) + ",\"window\":" + std::to_string(window) +
                ",\"display\":\"" + DisplayString(display) + "\",\"width\":" + std::to_string(width) +
                ",\"height\":" + std::to_string(height) + ",\"damage_percent\":" + std::to_string(percent) +
                ",\"effective_damage_percent\":" + std::to_string(actual) + ",\"object_width\":" + std::to_string(objectWidth) +
                ",\"object_height\":" + std::to_string(objectHeight) + ",\"step\":2,\"night\":" + (night ? "true" : "false") +
                ",\"frames\":" + std::to_string(frames) + "}\n";
        };
        configure();
        if (!ready.empty()) {
            std::ofstream file(ready + ".tmp"); file << status(); file.close();
            if (!file || rename((ready + ".tmp").c_str(), ready.c_str())) throw std::runtime_error("cannot write readiness file");
        }
        fputs(status().c_str(), stdout); fflush(stdout);
        auto start = std::chrono::steady_clock::now();
        auto next = start;
        while (running && std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < duration) {
            if (server >= 0) {
                int fd = accept4(server, nullptr, nullptr, SOCK_CLOEXEC);
                if (fd >= 0) {
                    timeval timeout{1, 0}; setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
                    char input[128]{}; int count = recv(fd, input, sizeof(input) - 1, 0);
                    std::string reply;
                    if (count > 0) {
                        double value; int enabled;
                        if (sscanf(input, "state %lf %d", &value, &enabled) == 2 && std::isfinite(value) && value >= 0 && value <= 100 && (enabled == 0 || enabled == 1)) {
                            percent = value; night = enabled; configure(); reply = status();
                        } else if (!strncmp(input, "status", 6)) reply = status();
                        else if (!strncmp(input, "quit", 4)) { running = 0; reply = status(); }
                        else reply = "{\"error\":\"invalid command\"}\n";
                    }
                    if (!reply.empty()) send(fd, reply.data(), reply.size(), MSG_NOSIGNAL);
                    close(fd);
                }
            }
            while (XPending(display)) {
                XEvent event; XNextEvent(display, &event);
                if (event.type == DestroyNotify) running = 0;
            }
            if (percent > 0 && running) {
                int oldX = x, oldY = y;
                if (percent != 100) {
                    if (x + dx < 0 || x + dx + objectWidth > width) dx = -dx;
                    if (y + dy < 0 || y + dy + objectHeight > height) dy = -dy;
                    x += dx; y += dy;
                }
                int left = std::min(x, oldX), top = std::min(y, oldY);
                XSetForeground(display, gc, 0x102030);
                XFillRectangle(display, window, gc, left, top, objectWidth + abs(x - oldX), objectHeight + abs(y - oldY));
                // Change the object every frame, including the overlapping footprint.
                unsigned long color = 0x204080 ^ ((frames & 1) ? 0x80b040 : 0);
                XSetForeground(display, gc, color);
                XFillRectangle(display, window, gc, x, y, objectWidth, objectHeight);
                XSync(display, False);
                ++frames;
            }
            next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1 / fps));
            if (next < std::chrono::steady_clock::now()) next = std::chrono::steady_clock::now();
            std::this_thread::sleep_until(next);
        }
        XFreeGC(display, gc); XFreeCursor(display, cursor); XDestroyWindow(display, window); XCloseDisplay(display); display = nullptr;
        if (server >= 0) close(server);
        if (!control.empty()) unlink(control.c_str());
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "gamescope-damage: %s\n", e.what());
        if (display) XCloseDisplay(display);
        if (server >= 0) { close(server); unlink(control.c_str()); }
        return 1;
    }
}
