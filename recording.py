import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
import grant_access_server as server
import realsense_filters as filters

default_depth_resolution_for_D400 = (1280, 720)
default_color_resolution_for_D400 = (1920, 1080)
default_depth_resolution_for_L500 = (320, 240)  # (1024, 768) #
default_color_resolution_for_L500 = (1920, 1080)


def configure_streams(config):
    device_name = device_connected()
    # change_FPS(config, 30)
    if device_name:
        if device_name == "D400":
            config.enable_stream(rs.stream.depth, *default_depth_resolution_for_D400, framerate=30)
            config.enable_stream(rs.stream.color, *default_color_resolution_for_D400, framerate=30)
        if device_name == "L500":
            config.enable_stream(rs.stream.depth, *default_depth_resolution_for_L500, framerate=30)
            config.enable_stream(rs.stream.color, *default_color_resolution_for_L500, framerate=30)
    return config


def wait_for_exposure_adaption(pipeline, waiting_frames=10):
    for i in range(waiting_frames):
        pipeline.wait_for_frames()
    return pipeline


def change_FPS(config, desired_fps):  ## Doesn't work!!
    config.disable_all_streams()
    config.enable_stream(rs.stream.depth, 0, rs.format.z16, desired_fps)
    config.enable_stream(rs.stream.color, 0, rs.format.rgb8, desired_fps)
    return config


def device_connected():
    if len(rs.context().devices) == 0:
        return False
    else:
        return rs.context().devices[0].get_info(rs.camera_info.product_line)


def start_recording(filename: str, pipeline=None, repeat_playback=None,
                    realtime=None):  # Realtime and repeat_playback are not used, just for consistency with "start_playback"
    if os.path.split(filename)[0] != '':
        os.makedirs(os.path.join(*os.path.split(filename)[:-1]), exist_ok=True)
    if pipeline:
        device = pipeline.get_active_profile().get_device()
        if not device.is_recorder():
            pipeline.stop()
            config = rs.config()
            configure_streams(config)
            config.enable_record_to_file(filename)
            pipeline.start(config)
            device = pipeline.get_active_profile().get_device()
        else:
            device.as_recorder().resume()
    else:
        pipeline = rs.pipeline()
        config = rs.config()
        configure_streams(config)
        config.enable_record_to_file(filename)
        pipeline.start(config)

    return pipeline


def pause_recording(pipeline):
    # if device_connected():
    device = pipeline.get_active_profile().get_device()
    device.as_recorder().pause()
    return pipeline


def resume_recording(pipeline):
    # if device_connected():
    device = pipeline.get_active_profile().get_device()
    device.as_recorder().resume()
    return pipeline


def stop_recording(pipeline):
    pipeline.stop()
    # pipeline = rs.pipeline()
    if device_connected():
        pipeline.start()
    return pipeline


def start_playback(filename: str, pipeline=None, repeat_playback=False, realtime=False):
    if pipeline:
        has_to_stop_device = False
        try:
            device = pipeline.get_active_profile().get_device()
            if device.is_playback():
                pipeline.stop()
                has_to_stop_device = True
        except:
            has_to_stop_device = True

        if has_to_stop_device:
            # pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device_from_file(filename, repeat_playback=repeat_playback)
            pipeline.start(config)
            device = pipeline.get_active_profile().get_device()
        else:
            device.as_playback().resume()
    else:
        pipeline = rs.pipeline()
        config = rs.config()
        configure_streams(config)
        config.enable_device_from_file(filename, repeat_playback=repeat_playback)
        pipeline.start(config)
    if not realtime:
        pipeline.get_active_profile().get_device().as_playback().set_real_time(realtime)

    return pipeline


def pause_playback(pipeline):
    # if device_connected():
    device = pipeline.get_active_profile().get_device()
    device.as_playback().pause()
    return pipeline


def resume_playback(pipeline):
    # if device_connected():
    device = pipeline.get_active_profile().get_device()
    device.as_playback().resume()
    return pipeline


def stop_playback(pipeline, restart_device = False):
    pipeline.stop()
    # pipeline = rs.pipeline()
    if device_connected() and restart_device:
        pipeline.start()
    return pipeline


def get_frame_timestamp(composite_frame):
    if isinstance(composite_frame, rs.composite_frame):
        return composite_frame.get_depth_frame().get_frame_metadata(rs.frame_metadata_value.time_of_arrival)
    elif isinstance(composite_frame, rs.frame):
        return composite_frame.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)
    else:
        raise Exception(f"Wrong type given to get timestamp of image. "
                        f"Given: {type(composite_frame)}, wanted: rs.composite_frame or rs.frame")


if __name__ == "__main__":

    # Example of usage: python recording.py --play -f C:/Users/march/Desktop/recordings/walking_slow.bag
    # or
    # python recording.py --play -f \\lst-data2.local.univ-savoie.fr\iot\data\realsense_d435i_raw\second_sequence_on_the_server.bag
    import argparse

    parser = argparse.ArgumentParser(description='Capture a sequence or play a recorded sequence')
    play_mode = parser.add_mutually_exclusive_group(required=True)
    play_mode.add_argument('--just-show', dest='just_show',
                           help='Just display the stream of the camera', action='store_true')
    play_mode.add_argument('-f', '--filename', dest='filename', metavar='filename', type=str,
                           help='Filename where you want to store/play the sequence.' +
                                '\nAvoid saving on removable devices to increase the bitrate')
    play_mode.add_argument('-d', '--directory', dest='directory', metavar='directory', type=str,
                           help='Directory where you want to save the sequence. File name will be timestamped.' +
                                '\nAvoid saving on removable devices to increase the bitrate')
    parser.add_argument('--record', dest='recording', help='Record and save the sequence', action='store_true')
    parser.add_argument('--play', dest='play', help='Play the sequence', action='store_true')
    parser.add_argument('--duration', dest='duration', type=int,
                        help='Set the recording for a certain amount of time (in seconds)')
    parser.add_argument('--continuous', dest='continuous', action='store_true',
                        help='Repeat recordings continuously')

    args = parser.parse_args()
    filename = args.filename
    directory = args.directory
    recording = args.recording
    just_show = args.just_show
    duration = args.duration
    continuous = args.continuous

    if filename is not None:
        if os.path.pathsep * 2 in filename:  # path contains "\\"
            if not server.grant_access_server():  # Create connection with the server
                print("No connection with the server, maybe turn on the VPN?")

    repeat_recording = True

    pipeline = rs.pipeline()
    pipeline.start()

    while repeat_recording:
        if (filename is None and directory is not None) or continuous:
            filename = directory + "/" + time.strftime(
                "%Y-%m-%d_%H_%M_%S_" + str(int((time.time() - int(time.time())) * 1000)) + ".bag")

        if just_show:
            pipeline = rs.pipeline()
            config = rs.config()
            configure_streams(config)
            pipeline.start(config)
            print("Start playing")
        elif recording:
            pipeline = start_recording(filename, realtime=True, pipeline=pipeline)
            print("Recording started")
        else:
            pipeline = start_playback(filename, realtime=True)
            print("Playback started")

        frame_count = 0

        start_time = time.time()

        align = rs.align(rs.stream.color)

        currently_playing = True
        while True:
            try:
                # This call waits until a new coherent set of frames is available on a device
                # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
                if currently_playing:
                    frames = pipeline.wait_for_frames()
                    frames.keep()
                    frame_count += 1
                frames = align.process(frames)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                depth_frame = rs.colorizer().colorize(
                    # frames.get_depth_frame()).get_data()
                    # filters.aruco_detection_filters(frames.get_depth_frame())).get_data()
                    filters.l515_filter(frames.get_depth_frame())).get_data()
                cv2.imshow('depth', cv2.resize(np.array(depth_frame), (800, 600)))
                # cv2.imshow('depth', cv2.cvtColor(np.asanyarray(color_frame.get_data()).copy(), cv2.COLOR_BGR2RGB))

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                if key == ord('p'):
                    if not just_show:
                        if currently_playing:
                            if recording:
                                pipeline = pause_recording(pipeline)
                            else:
                                pipeline = pause_playback(pipeline)
                        else:
                            if recording:
                                pipeline = resume_recording(pipeline)
                            else:
                                pipeline = resume_playback(pipeline)
                        currently_playing = not currently_playing
            except Exception as e:
                print(e)
                break
            if duration is not None and time.time() > start_time + duration:
                break

        sequence_duration = time.time() - start_time
        start_time = time.time()
        if just_show:
            pipeline.stop()
        elif recording:
            print("Recording ended, saving file")
            pipeline = stop_recording(pipeline)
            saving_duration = time.time() - start_time
            print(f"Video duration : {round(sequence_duration, 3)}s")
            print(f"Saving duration: {round(saving_duration, 3)}s")
        else:
            pipeline = stop_playback(pipeline)

        if continuous:
            repeat_recording = True
        else:
            repeat_recording = False

    print(" --- End ---")
