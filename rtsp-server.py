import subprocess
import shlex
import socket
import sys
import os
import time

def get_local_ip():
    """
    Returns the primary local IP address (non-127.0.0.1)
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't have to be reachable; just used to pick a local interface
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def main():
    device = "/dev/video0"   # change if your webcam is different
    port = 8554
    stream_name = "webcam"

    server_ip = get_local_ip()
    rtsp_url = f"rtsp://{server_ip}:{port}/{stream_name}"

    print("===============================================")
    print(f"Using webcam device: {device}")
    print(f"RTSP URL for clients: {rtsp_url}")
    print("Press Ctrl+C to stop.")
    print("===============================================")

    # Use ffmpeg to stream to RTSP via TCP with h264
    ffmpeg_cmd = f"""
        ffmpeg
        -f v4l2 -framerate 30 -video_size 1280x720 -input_format yuyv422 -i {device}
        -vcodec libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -b:v 2M
        -f rtsp -rtsp_transport tcp rtsp://localhost:{port}/{stream_name}
    """

    # Clean up whitespace
    ffmpeg_cmd = " ".join(ffmpeg_cmd.split())
    print("Running FFmpeg command:")
    print(ffmpeg_cmd)
    print()

    proc = None
    try:
        proc = subprocess.Popen(shlex.split(ffmpeg_cmd))
        proc.wait()
    except KeyboardInterrupt:
        print("\nStopping RTSP server...")
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    except Exception as e:
        print("Error while running ffmpeg:", e)
        if proc:
            try:
                proc.terminate()
            except Exception:
                pass
        sys.exit(1)

if __name__ == "__main__":
    main()

