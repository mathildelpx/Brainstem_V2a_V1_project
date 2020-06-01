from PIL import Image, ImageSequence, ImageOps
import pandas as pd
import cv2
import numpy as np
import imageio
import math


def resize(tif_image, factor):
    """Resize PIL Image object, by a defined factor, applying the same resizing on all axes.
    Return the resized image."""
    for image in ImageSequence.Iterator(tif_image):
        width_pixel = image.width//factor
        height_pixel = image.height//factor
        image.resize((width_pixel, height_pixel))
    return tif_image


def cut_calcium_imaging(bout, df_bouts, ca_tiff, fq, fps, output_path, analysis_log):
    """For a specified bout, cut the calcium imagign video to start from the bout start
    (take the lower round frame)
    and 3 seconds after the start.
    Uses TiffImage obeject from PIL, creates a list of all the subframes to save, and save it as gif with
    specified frame rate."""
    bin_factor = 2
    start = math.floor((df_bouts.BoutStartVideo[bout]/fq)*fps)
    end = start + int(2*fps)
    frames = np.arange(start, end+1)
    substack = []
    for i, frame in enumerate(ImageSequence.Iterator(ca_tiff)):
        if frames in frames:
            frame = frame.convert('L')
            frame = ImageOps.autocontrast(frame)
            width = frame.width // bin_factor
            height = frame.height // bin_factor
            frame = frame.resize((width, height), Image.ANTIALIAS)
            frame = np.asarray(frame)
            substack.append(frame)
    # works and is openable but very very heavy
    imageio.mimsave(output_path+ str(bout) + '/calcium_imaging.gif', substack, duration=1/fps)
    # for avi saving, for now works and is lighter, but not openable by imageJ
    imageio.mimsave(output_path + str(bout) + '/calcium_imaging.avi', substack, format='avi')
    print('successfully saved CI gif for bout', bout, 'start:', start, 'end', end)
    analysis_log.loc[bout, 'Start_CI_frame'] = start
    analysis_log.loc[bout, 'End_CI_frame'] = end
    analysis_log.loc[bout, 'Start_CI_time'] = start/fps
    analysis_log.loc[bout, 'End_CI_time'] = end/fps


def cut_behavior(bout, video, fps, output_path, analysis_log):
    """For a specified bout, cut the behavior video from the bout start to 3 seconds after the start.
    Uses video reader from imageio object, creates a list of all the subframes to save, and save it as tif."""
    start = int(analysis_log.loc[bout, 'Start_CI_time']*fps)
    end = int(analysis_log.loc[bout, 'End_CI_time']*fps)
    behavior_substack=[]
    for i, im in enumerate(video):
        if i in np.arange(start, end+1):
            behavior_substack.append(im)
    imageio.mimsave(output_path+str(bout)+'/behavior.tif', behavior_substack)
    print('successfully saved behavior tif for bout', bout)
    print('start:', start/fps, 's (', start, ') , end:', end/fps, 's (', end, ')')
    analysis_log.loc[bout, 'Start_beh_frame'] = start
    analysis_log.loc[bout, 'End_beh_frame'] = end
    analysis_log.loc[bout, 'Start_beh_time'] = start/fps
    analysis_log.loc[bout, 'End_beh_time'] = end/fps


def resize_condense_behavior(bout, fps, output_path):
    """For a specified bout, load the tif created by cut_behavior, using PIL Image.open function.
    Resize the video by dividing the pixels size by 4 on each axis, and convert into grayscale.
    Save the resulting images as gif, with a specified frame rate"""
    tif_behavior = Image.open(output_path+str(bout)+'/behavior.tif')
    resized_behavior = resize(tif_behavior, 8)
    final_behavior = []
    for image in ImageSequence.Iterator(resized_behavior):
        # convert to grey scale
        image = image.convert('L')
        image = ImageOps.invert(image)
        image = ImageOps.crop(image, 50)
        final_behavior.append(image)
    imageio.mimsave(output_path + str(bout) + '/behavior.gif', final_behavior, duration=1/fps)
    print('successfully saved behavior gif for bout', bout)


def createShorterVideo(inputVideoName, outputVideoName, startFrame, endFrame):
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(inputVideoName)

    # Check if input video is opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(outputVideoName, cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width, frame_height))

    i = startFrame
    maxx = endFrame

    cap.set(1, i)
    # Read until video is completed
    while (cap.isOpened() and (i < maxx)):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            out.write(frame)
        # Break the loop
        else:
            break

        i = i + 1

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
