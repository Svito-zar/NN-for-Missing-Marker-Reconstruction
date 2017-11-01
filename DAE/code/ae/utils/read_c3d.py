import btk
import numpy as np
from flags import *

def read_c3d_file(file_name):
    """
    Reads a file from CMU MoCap dataset in the c3d format

    Args:
        filename - adress of the file with the MoCap data in the c3d format
    Returns:
        sequence - array T * ND

    """

    # First read a set of marker labels, in order to filter out unnasesery one
    labels_file = open("MainLabels.txt", "r")


    # Read the data
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(file_name)  # set a filename to the reader
    reader.Update()
    acq = reader.GetOutput()  # acq is the btk aquisition object

    #print 'Reading a c3d file', file_name

    all_3d_coords = np.array([])
    point_id = 0

    # Get subject name
    label_string = acq.GetPoint(0).GetLabel()
    name = label_string.split(':', 2)[0]

    while (True):
        try:
            # Get the next label
            label = name + ':' + labels_file.readline().splitlines()[0]
            next_point = acq.GetPoint(label).GetValues()
        except IndexError:
            #print 'Read', point_id, 'skeleton 3d points during', acq.GetPointFrameNumber(), 'time frames'
            break

        # Concatanate curr chunks to all of them
        all_3d_coords = np.dstack([all_3d_coords, next_point]) if all_3d_coords.size else np.array(next_point)

        point_id += 1

    # Make a proper array shape
    mocap_seq = all_3d_coords.reshape(all_3d_coords.shape[0], -1)

    return mocap_seq


def read_a_folder(curr_dir):
    chunk_length = FLAGS.chunk_length
    stride = FLAGS.chunking_stride

    data = np.array([])

    for filename in os.listdir(curr_dir):
        curr_sequence = read_c3d_file(curr_dir + '/' + filename)

        curr_chunks = np.array([curr_sequence[i:i + chunk_length, :] for i in
                                xrange(0, len(curr_sequence) - chunk_length, stride)])  # Split sequence into chunks
        if (curr_chunks.shape[0] > 0):
            # Concatanate curr chunks to all of them
            data = np.vstack([data, curr_chunks]) if data.size else np.array(curr_chunks)

    data = np.array(data)

    return data


if __name__ == '__main__':
    print(FLAGS.data_dir)
    dataset = read_a_folder(FLAGS.data_dir+'/Train')
    print 'The dataset size is ', dataset.shape
    #read_file('/home/taras/Downloads/06_05.c3d')