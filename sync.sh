#!/bin/sh

usage() {
    echo "Usage: $0 [--arch native|arm] [--appid ID] <host>"
    echo "  ARM builds use aarch64-linux-gnu tools and AARCH64_SYSROOT"
    echo "  (default: build-arm/sysroot if present, otherwise /usr/aarch64-linux-gnu/sys-root)."
}

HOST=
APPID=
ARCH=native

while [ "$#" -gt 0 ]; do
    case "$1" in
        --arch)
            if [ "$#" -lt 2 ]; then
                echo "--arch requires native or arm" >&2
                usage >&2
                exit 1
            fi
            ARCH=$2
            shift 2
            ;;
        --appid)
            if [ "$#" -lt 2 ]; then
                echo "--appid requires a Steam application ID" >&2
                usage >&2
                exit 1
            fi
            APPID=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            if [ -n "$HOST" ]; then
                echo "Only one host may be specified" >&2
                usage >&2
                exit 1
            fi
            HOST=$1
            shift
            ;;
    esac
done

if [ -z "$HOST" ]; then
    usage >&2
    exit 1
fi

case "$APPID" in
    "") ;;
    0|*[!0-9]*)
        echo "Invalid Steam application ID: $APPID" >&2
        exit 1
        ;;
esac

set -e

case "$ARCH" in
    native)
        TARGET_ARCH=$(uname -m)
        BUILD_DIR=build
        set --
        ;;
    arm|aarch64|arm64)
        TARGET_ARCH=aarch64
        BUILD_DIR=build-arm
        if [ -d "$BUILD_DIR/sysroot/usr/include" ]; then
            AARCH64_SYSROOT=${AARCH64_SYSROOT:-$BUILD_DIR/sysroot}
        else
            AARCH64_SYSROOT=${AARCH64_SYSROOT:-/usr/aarch64-linux-gnu/sys-root}
        fi
        if [ ! -d "$AARCH64_SYSROOT/usr/include" ] || [ ! -d "$AARCH64_SYSROOT/usr/lib64" ]; then
            echo "ARM64 sysroot is missing development headers or libraries: $AARCH64_SYSROOT" >&2
            echo "Set AARCH64_SYSROOT to an ARM64 sysroot containing Gamescope's build dependencies." >&2
            exit 1
        fi
        AARCH64_SYSROOT=$(cd "$AARCH64_SYSROOT" && pwd)
        for tool in aarch64-linux-gnu-gcc aarch64-linux-gnu-g++ aarch64-linux-gnu-ar aarch64-linux-gnu-strip pkg-config python3; do
            if ! command -v "$tool" >/dev/null 2>&1; then
                echo "Required ARM64 build tool not found: $tool" >&2
                exit 1
            fi
        done
        mkdir -p "$BUILD_DIR"
        python3 - "$AARCH64_SYSROOT" "$BUILD_DIR/cross.ini" "$(aarch64-linux-gnu-g++ -dumpversion)" <<'PY'
from pathlib import Path
import sys

sysroot, output, cpp_version = sys.argv[1:]
def quote(value):
    if "'''" in value or '\n' in value or '\r' in value:
        sys.exit('Sysroot path cannot contain newlines or three consecutive apostrophes')
    return "'''" + value + "'''"

cpp_include = Path(sysroot) / 'usr/include/c++' / cpp_version
cpp_configs = list(cpp_include.glob('*/bits/c++config.h'))
if not cpp_configs:
    sys.exit(f'ARM64 libstdc++ development headers for GCC {cpp_version} are missing from {sysroot}')
c_args = ['--sysroot=' + sysroot]
cpp_args = c_args.copy()
for directory in (cpp_include, cpp_configs[0].parent.parent, cpp_include / 'backward'):
    cpp_args += ['-isystem', str(directory)]
link_args = ['--sysroot=' + sysroot]
gcc_libdir = Path(sysroot) / 'usr/lib/gcc' / cpp_configs[0].parent.parent.name / cpp_version
if gcc_libdir.is_dir():
    link_args += ['-L' + str(gcc_libdir)]
    system_includes = ['-nostdinc', '-isystem', str(gcc_libdir / 'include')]
    if (gcc_libdir / 'include-fixed').is_dir():
        system_includes += ['-isystem', str(gcc_libdir / 'include-fixed')]
    system_includes += ['-isystem', str(Path(sysroot) / 'usr/include')]
    c_args += system_includes
    cpp_args += system_includes
cpp_link_args = link_args + ['-lgcc_s']

Path(output).write_text(f"""[binaries]
c = 'aarch64-linux-gnu-gcc'
cpp = 'aarch64-linux-gnu-g++'
ar = 'aarch64-linux-gnu-ar'
strip = 'aarch64-linux-gnu-strip'
pkg-config = ['env', 'PKG_CONFIG_FDO_SYSROOT_RULES=1', 'pkg-config']

[host_machine]
system = 'linux'
cpu_family = 'aarch64'
cpu = 'armv8-a'
endian = 'little'

[properties]
needs_exe_wrapper = true
sys_root = {quote(sysroot)}
pkg_config_libdir = [{quote(sysroot + '/usr/lib64/pkgconfig')}, {quote(sysroot + '/usr/share/pkgconfig')}]

[built-in options]
c_args = [{', '.join(map(quote, c_args))}]
cpp_args = [{', '.join(map(quote, cpp_args))}]
c_link_args = [{', '.join(map(quote, link_args))}]
cpp_link_args = [{', '.join(map(quote, cpp_link_args))}]
""")
PY
        set -- --cross-file "$BUILD_DIR/cross.ini"
        unset PKG_CONFIG_PATH PKG_CONFIG_LIBDIR PKG_CONFIG_SYSROOT_DIR
        export PKG_CONFIG_FDO_SYSROOT_RULES=1
        ;;
    *)
        echo "Unsupported architecture: $ARCH (expected native or arm)" >&2
        exit 1
        ;;
esac

REMOTE_ARCH=$(ssh "$HOST" uname -m)
if [ "$REMOTE_ARCH" != "$TARGET_ARCH" ]; then
    echo "Target architecture $TARGET_ARCH does not match $HOST ($REMOTE_ARCH). Use --arch arm for ARM64." >&2
    exit 1
fi

meson setup "$BUILD_DIR" "$@" --prefix=/usr --libdir=lib64 -Dforce_fallback_for=stb,libdisplay-info,libliftoff,wlroots,vkroots -Denable_openvr_support=false
ninja -C "$BUILD_DIR"
scp "$BUILD_DIR/src/gamescope" "${HOST}:gamescope"
scp "$BUILD_DIR/layer/libVkLayer_FROG_gamescope_wsi_${TARGET_ARCH}.so" "${HOST}:gamescope-wsi.so"

ssh "$HOST" /bin/bash -s -- "${APPID:-0}" "$TARGET_ARCH" <<'EOF'
    set -e

    appid=$1
    if [ "$appid" = 0 ]; then
        appid=
    fi
    target_arch=$2
    wsi_library="libVkLayer_FROG_gamescope_wsi_${target_arch}.so"

    login_config_mounted=
    if mountpoint -q /etc/plasmalogin.conf; then
        login_config_mounted=1
    fi
    session_user=$(id -un)

    passwd_entry=$(getent passwd "$session_user") || {
        echo "Unknown remote user: $session_user" >&2
        exit 1
    }
    session_home=$(printf '%s\n' "$passwd_entry" | cut -d: -f6)
    steam_pipe="$session_home/.steam/steam.pipe"
    old_steam_pids=
    if [ -n "$appid" ]; then
        old_steam_pids=$(lsof -t "$steam_pipe" 2>/dev/null || true)
    fi

    if ! sudo test -w /usr/bin; then
        sudo rpm-ostree usroverlay || true
    fi
    if [ -z "$login_config_mounted" ]; then
        printf '%s\n' \
            '[Autologin]' \
            'Relogin=true' \
            "User=$session_user" \
            'Session=gamemode.desktop' |
            sudo tee /run/plasmalogin.conf >/dev/null
        sudo chcon --reference=/etc/plasmalogin.conf /run/plasmalogin.conf
        sudo mount --bind /run/plasmalogin.conf /etc/plasmalogin.conf
    fi
    sudo install -Z -m755 ~/gamescope /usr/bin/gamescope
    sudo install -Z -m755 ~/gamescope-wsi.so "/usr/lib64/$wsi_library"
    flatpak_platform_override="/run/anatase/flatpak/overrides/org.anatase.Platform/$target_arch/stable"
    if [ -d "$flatpak_platform_override" ]; then
        sudo install -Dm755 "/usr/lib64/$wsi_library" \
            "$flatpak_platform_override/lib64/$wsi_library"
    fi
    flatpak_runtime=$(readlink -f "$(flatpak info --arch="$target_arch" -l org.anatase.Platform 2>/dev/null || true)")
    if [ -n "$flatpak_runtime" ]; then
        flatpak_wsi="$flatpak_runtime/files/lib64/$wsi_library"
        if [ -f "$flatpak_wsi" ]; then
            if mountpoint -q "$flatpak_wsi"; then
                sudo umount -l "$flatpak_wsi"
            fi
            sudo mount --bind "$session_home/gamescope-wsi.so" "$flatpak_wsi"
        fi
    fi
    sudo setcap cap_sys_nice=ep /usr/bin/gamescope
    sudo systemctl stop plasmalogin
    sleep 2
    sudo systemctl start plasmalogin

    if [ -n "$appid" ]; then
        attempts=0
        while :; do
            steam_pid=$(lsof -t "$steam_pipe" 2>/dev/null | head -n 1 || true)
            if [ -n "$steam_pid" ] && ! printf '%s\n' "$old_steam_pids" | grep -Fxq "$steam_pid"; then
                break
            fi

            attempts=$((attempts + 1))
            if [ "$attempts" -ge 120 ]; then
                echo "Timed out waiting for Steam to accept launch requests" >&2
                exit 1
            fi
            sleep 0.5
        done

        game_uri="steam://rungameid/$appid"
        if ! timeout 5 sh -c 'printf "%s\n" "$1" > "$2"' sh "$game_uri" "$steam_pipe"; then
            echo "Failed to send Steam launch request for app $appid" >&2
            exit 1
        fi
        echo "Requested Steam app $appid"
    fi
EOF
