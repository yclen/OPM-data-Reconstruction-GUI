import tkinter as tk
from tkinter import filedialog
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndi
import tifffile as tf
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import time
import os
import napari

#set parameters
y_step_size = 0.347 #um
pixelsize = 0.17334 #um

bin = 1
slices_per_volume = "z"
channels = 1




def get_transformed_shape(shape, matrix):
    # Create an array of all 8 corners of the 3D shape
    corners = np.array([
        [0, 0, 0, 1],
        [shape[0], 0, 0, 1],
        [0, shape[1], 0, 1],
        [0, 0, shape[2], 1],
        [shape[0], shape[1], 0, 1],
        [shape[0], 0, shape[2], 1],
        [0, shape[1], shape[2], 1],
        [shape[0], shape[1], shape[2], 1]
    ])

    # Apply the transformation matrix to all corners
    transformed_corners = np.round(matrix @ corners.T).astype(int).T

    # Find the new bounding box
    min_corner = np.min(transformed_corners, axis=0)
    max_corner = np.max(transformed_corners, axis=0)

    # Calculate the dimensions of the transformed shape
    new_shape = max_corner - min_corner

    return tuple(new_shape[:3])
def all_in_one_matrix(shape, skewfactor, scalefactor, angle):
    #define matrix functions
    def rotate_around_x(angle):

        matrix = np.eye(4)
        matrix[0,0] = np.cos(angle*np.pi/180.0)
        matrix[1,1] = np.cos(angle*np.pi/180.0)
        matrix[1,0] = np.sin(angle*np.pi/180.0)
        matrix[0,1] = -np.sin(angle*np.pi/180.0)

        return matrix

    def shift_center(shape):
        matrix = np.eye(4)

        matrix[0,3] = -shape[0]/2
        matrix[1,3] = -shape[1]/2
        matrix[2,3] = -shape[2]/2

        return matrix

    def unshift_center(shape):
        matrix = np.eye(4)

        matrix[0,3] = shape[0]/2
        matrix[1,3] = shape[1]/2
        matrix[2,3] = shape[2]/2

        return matrix

    #define matrices
    skew = np.eye(4)
    skew[1,0] = skewfactor

    scale = np.eye(4)
    scale[0,0] = scalefactor

    rotate = rotate_around_x(angle)

    shift = shift_center(shape)
    output_shape = get_transformed_shape(shape, rotate@scale@skew@shift)
    unshift = unshift_center(output_shape)

    return unshift@rotate@scale@skew@shift


class transformer:

    angle = 135
    interpolation = 1
    images = ()
    max_GPU_memory = 1 #GB

    def __init__(self, y_step_size, pixelsize=0.08667):
        self.y_step_size = y_step_size
        self.pixelsize = pixelsize
        self.skewfactor = np.cos(self.angle*np.pi/180.0)*y_step_size/pixelsize
        self.scalefactor = np.sin(self.angle*np.pi/180.0)*y_step_size/pixelsize
        self.sep_or_tog = "together"
        self.output = np.asarray([1])


    def read_file(self, path):
        print("reading file...")
        images = ()
        with tf.TiffFile(path) as tif:
            for i, page in enumerate(tif.pages):
                try:
                    image = page.asarray()
                    # print("Loading image:",i)
                    images = images + (image,)
                except Exception as e:
                    print(f"Error reading page {i}: {e}")

        self.vol = np.stack(images, axis=0)
        self.shape = self.vol.shape
        self.dsize = round(self.vol.nbytes/1024**3, 5) #GB
        print("read succesfully.")

    def print_info(self):
        self.shape = self.vol.shape
        self.dsize = round(self.vol.nbytes/1024**3, 5) #GB
        print("volume shape:",self.shape,"data size:",self.dsize,"GB")
    def print_T_info(self, num_time_points):
        t_shape = get_transformed_shape(self.shape[-3:], self.matrix)
        t_datasize = round(2*t_shape[0]/1024**3*t_shape[1]*t_shape[2],5)

        # print("transformed datasize of each volume:",t_datasize,"GB","\nTotal final datasize:",num_time_points*t_datasize, "GB")

    def view_image(self, original_volume, images=()):
        viewer = napari.view_image(images[0], name="image number 1")
        for i in range(len(images)-1):
            if len(images)>1:
                image_layer = viewer.add_image(images[i+1], name=f"image number {i+2}")

        print("opening viewer...")
        napari.run()
        print("closed viewer.")

    def GPU_transform(self, vol):
        matrix = self.matrix
        vol_cp = cp.asarray(vol)
        matrix_cp = cp.asarray(matrix)
        output_shape = get_transformed_shape(vol_cp.shape, matrix)
        output_cp = ndi.affine_transform(vol_cp, cp.linalg.inv(matrix_cp), output_shape=output_shape, order=self.interpolation)
        return output_cp.get()

    def GPU_transform_seperate(self, vol):

            matrix = self.matrix
            interpolation = self.interpolation
            #define matrix functions
            def rotate_around_x(angle):

                matrix = np.eye(4)
                matrix[0,0] = np.cos(angle*np.pi/180.0)
                matrix[1,1] = np.cos(angle*np.pi/180.0)
                matrix[1,0] = np.sin(angle*np.pi/180.0)
                matrix[0,1] = -np.sin(angle*np.pi/180.0)

                return matrix
            def shift_center(shape):
                matrix = np.eye(4)

                matrix[0,3] = -shape[0]/2
                matrix[1,3] = -shape[1]/2
                matrix[2,3] = -shape[2]/2

                return matrix
            def unshift_center(shape):
                matrix = np.eye(4)

                matrix[0,3] = shape[0]/2
                matrix[1,3] = shape[1]/2
                matrix[2,3] = shape[2]/2

                return matrix

            #define matrices
            skew = np.eye(4)
            skew[1,0] = self.skewfactor

            scale = np.eye(4)
            scale[0,0] = self.scalefactor

            rotate = rotate_around_x(self.angle)

            #calculate output shapes and cupy matracies
            # print("shape before:",vol.shape)
            skew_output_shape = get_transformed_shape(vol.shape, skew)
            # print("shape after skew:",skew_output_shape)
            scale_output_shape = get_transformed_shape(skew_output_shape, scale)
            # print("shape after scale:",scale_output_shape)

            shift = shift_center(scale_output_shape)
            output_shape = get_transformed_shape(scale_output_shape, rotate@shift)
            unshift = unshift_center(output_shape)


            tru_t_shape = get_transformed_shape(vol.shape, matrix)


            rotate_output_shape = get_transformed_shape(scale_output_shape, unshift@rotate@shift)
            # print("shape after rotate:",rotate_output_shape)

            #preform all GPU transformations
            vol_cp = cp.asarray(vol)
            vol_cp = ndi.affine_transform(vol_cp, cp.linalg.inv(cp.asarray(skew)), output_shape=skew_output_shape, order=interpolation)
            vol_cp = ndi.affine_transform(vol_cp, cp.linalg.inv(cp.asarray(scale)), output_shape=scale_output_shape, order=interpolation)
            vol_cp = ndi.affine_transform(vol_cp, cp.linalg.inv(cp.asarray(unshift@rotate@shift)), output_shape=(rotate_output_shape[0],rotate_output_shape[1],vol.shape[2]), order=interpolation)

            vol_cp = vol_cp[(vol_cp.shape[0]-tru_t_shape[0])/2:(vol_cp.shape[0]+tru_t_shape[0])/2,:, :]

            return vol_cp.get()

    def GPU_transform_by_Xslices(self, vol, how_transform="together"):
        matrix = self.matrix
        t_shape = get_transformed_shape(vol.shape, matrix)

        if how_transform=="together":
            t_function = self.GPU_transform
            self.max_GPU_memory = 1
        elif how_transform=="seperate":
            t_function = self.GPU_transform_seperate
            self.max_GPU_memory = .3

        #this represents the max number of bytes that can be stored in the GPU's ram at once (after transformation)
        max = self.max_GPU_memory*1024**3
        x = int(max/(2*(vol.shape[0]*vol.shape[1] + 2*t_shape[0]*t_shape[1])))

        #number of x-slices we need to make
        num_x = int(vol.shape[2]/x)
        images = (1,)

        # print("shape:",vol.shape,"num x slices:",num_x,"size of slice:",x,"remaining slices:",vol.shape[2]%x)
        #begin transforming slices
        for i in range(num_x):
            slice = t_function(vol[:,:,x*i:(i+1)*x])
            # print(f"shape of slice {i+1}:",slice.shape)
            images = images + (slice,)

        #transform final remaining slice
        final_slice = t_function(vol[:,:,x*num_x:])
        # print("shape of final slice:", final_slice.shape)
        images = images + (final_slice,)

        #combine all the slices
        output = np.concatenate(images[1:], axis=2)
        # print("complete stack:", output.shape)
        return output

    def GPU_timepoints(self, vol, slices_per_volume=None, how_transform="together"):
        if slices_per_volume is None:
            slices_per_volume = vol.shape[0]

        if len(vol.shape)==4:
            num_time_points = vol.shape[0]
        elif len(vol.shape)==3:
            if slices_per_volume=="z":
                slices_per_volume = vol.shape[0]

            slices_per_volume = int(slices_per_volume)
            num_time_points = int(vol.shape[0]/slices_per_volume)
            extra_slices = vol.shape[0]%slices_per_volume

        self.print_T_info(num_time_points)
        timepoints = (1,)
        for t in range(num_time_points):
            # print(f"Time Point number {t+1}")

            if len(vol.shape)==4:
                t_vol = self.GPU_transform_by_Xslices(vol[t,:,:,:], how_transform)
            elif len(vol.shape)==3:
                t_vol = self.GPU_transform_by_Xslices(vol[t*slices_per_volume:(t+1)*slices_per_volume,:,:], how_transform)
            timepoints = timepoints + (t_vol,)

        return np.stack(timepoints[1:], axis=0)

    def transform_big_files(self, vol, slices_per_volume):
        max = 50 #GB
        t_shape = get_transformed_shape(self.shape[-3:], self.matrix)
        t_datasize = round(2*t_shape[0]/1024**3*t_shape[1]*t_shape[2],5) #datasize per volume (GB)
        num_t_per_file = int(max/t_datasize)


        #calculate total number of timepoints
        if len(vol.shape)==4:
            num_time_points = vol.shape[0]
        elif len(vol.shape)==3:
            if slices_per_volume=="z":
                slices_per_volume = vol.shape[0]

            slices_per_volume = int(slices_per_volume)
            num_time_points = int(vol.shape[0]/slices_per_volume)
            extra_slices = vol.shape[0]%slices_per_volume


        # for f in range(int(num_time_points/num_t_per_file)):


        if len(vol.shape)==4:
            vol = vol[:num_t_per_file, :, :, :]
        elif len(vol.shape)==3:
            vol = vol[:num_t_per_file*slices_per_volume, :, :]

        output = self.GPU_timepoints(vol, slices_per_volume, self.sep_or_tog)
        tf.imwrite(save_location, output)

class gui_window:

    def __init__(self):
        window = tk.Tk()
        window.geometry("600x120")
        window.title("GPU deskewing of many files at once")
        self.paths = []


        #variables
        font1 = ('Arial',10, 'bold')

        #Y step size label
        Y_step_label = tk.Label(window, text="Y Step Size:           µm", font=font1)
        Y_step_label.place(x=5, y=10)
        self.Y_step_entry = tk.Entry(window, width=5, textvariable=tk.DoubleVar(value=y_step_size))
        self.Y_step_entry.place(x=90, y=11); d=40

        #slices per volume Entry
        slices_label = tk.Label(window, text="Slices per volume:", font=font1);s=0.3
        slices_label.place(x=190-d*s, y=10)
        optional_label = tk.Label(window, text="(may be optional)", font=('Arial',8))
        optional_label.place(x=200-d*s, y=30)
        self.slices_entry = tk.Entry(window, width=5, textvariable=tk.DoubleVar(value=slices_per_volume))
        self.slices_entry.place(x=315-d*s, y=11)

        #binning
        def select_bin(bined):
            T.pixelsize = 0.08667*bined
            print(T.pixelsize)
        binn = tk.IntVar()
        binn.set(bin)
        bin_menu = tk.OptionMenu(window, binn, 1,2,4, command=select_bin)
        bin_menu.place(x=500-2*d, y=8)
        bin_label = tk.Label(window, text="Binning:", font=font1)
        bin_label.place(x=440-2*d, y=10)

        #channels
        ch_label = tk.Label(window, text="CH:", font=font1); ch_label.place(x=500, y=10)
        self.ch_entry = tk.Entry(window, width=5, textvariable=tk.DoubleVar(value=channels))
        self.ch_entry.place(x=530,y=11)


        #select folder
        def choose_folder():
            path = filedialog.askdirectory()
            print(f"Selected folder: {path}")
            self.file_label.config(text=path)

            window.geometry("600x210")
            transform_button.place(x=10, y=120)
            s_or_t_operations_menu.place(x=140, y=165)
            s_or_t_label.place(x=5, y=170)

            tif_count = 0
            total_size = 0

            # List all files and filter for TIFF files
            for entry in os.scandir(path):
                if entry.is_file() and entry.name.lower().endswith(('.tif', '.tiff')):
                    self.paths.append(entry.path)
                    tif_count += 1
                    file_size = os.path.getsize(entry.path)
                    total_size += file_size
                    print(f"File {tif_count}: {entry.name}, Size: {round(file_size/1024**3,3)} Gb")

            print(f"Total number of TIFF files: {tif_count}")
            print(f"Total size of TIFF files: {round(total_size/1024**3,3)} Gb")

        button_choose = tk.Button(window, text="Choose folder", font=font1, command=choose_folder, bg="#C0C0C0")
        button_choose.place(x=10, y=70)

        #file label
        self.file_label = tk.Label(window, text="", font=('Arial',8, 'bold'), wraplength=500, justify=tk.LEFT)
        self.file_label.place(x=120, y=60)

        #transform button
        transform_button = tk.Button(window, text="Transform", font=('Arial',12, 'bold'), command=self.transform, bg="#C0C0C0")

        #seperate or together
        def select_s_or_t(s_or_t):
            T.sep_or_tog = s_or_t
            print(T.sep_or_tog)
        s_t = tk.StringVar()
        s_t.set("together")
        s_or_t_operations_menu = tk.OptionMenu(window, s_t, "together","seperate", command=select_s_or_t)
        s_or_t_label = tk.Label(window, text="Do all matrix operations:")






        window.mainloop()

    def transform(self):
        #create new folder
        recon_folder = os.path.join(self.file_label.cget("text"), "reconstructed_data")
        os.makedirs(recon_folder, exist_ok=True)

        for p in range(len(self.paths)):
            path = self.paths[p]
            print(f"transforming file number {p}:")

            T.read_file(path)
            # T.print_info()
            ch = int(self.ch_entry.get())


            T.y_step_size = float(self.Y_step_entry.get())
            T.skewfactor = np.cos(T.angle*np.pi/180.0)*T.y_step_size/T.pixelsize
            T.scalefactor = np.sin(T.angle*np.pi/180.0)*T.y_step_size/T.pixelsize
            T.matrix = all_in_one_matrix(T.shape[-3:], T.skewfactor, T.scalefactor, T.angle) #may need to be adjusted based on shape
            print("begining transformation...")

            if ch==2:
                if len(T.shape)==5:
                    if T.shape[1]<=2:
                        print("\nTransforming channel 1")
                        ch1 = T.GPU_timepoints(T.vol[:,0,:,:,:], self.slices_entry.get(), T.sep_or_tog)
                        print("\nTransforming channel 2")
                        ch2 = T.GPU_timepoints(T.vol[:,1,:,:,:], self.slices_entry.get(), T.sep_or_tog)
                        T.output = np.stack((ch1,ch2), axis=0)
                    elif T.shape[0]<=2:
                        print("\nTransforming channel 1")
                        ch1 = T.GPU_timepoints(T.vol[0,:,:,:,:], self.slices_entry.get(), T.sep_or_tog)
                        print("\nTransforming channel 2")
                        ch2 = T.GPU_timepoints(T.vol[1,:,:,:,:], self.slices_entry.get(), T.sep_or_tog)
                        T.output = np.stack((ch1,ch2), axis=0)
                    elif T.shape[2]<=2:
                        T.matrix = all_in_one_matrix(T.vol[:,:,0,:,:].shape[-3:], T.skewfactor, T.scalefactor, T.angle)
                        print("\nTransforming channel 1")
                        ch1 = T.GPU_timepoints(T.vol[:,:,0,:,:], self.slices_entry.get(), T.sep_or_tog)
                        print("\nTransforming channel 2")
                        ch2 = T.GPU_timepoints(T.vol[:,:,1,:,:], self.slices_entry.get(), T.sep_or_tog)
                        T.output = np.stack((ch1,ch2), axis=0)

                elif len(T.shape)==4:
                    if T.shape[0]<=2:
                        print("\nTransforming channel 1")
                        ch1 = T.GPU_timepoints(T.vol[0,:,:,:], self.slices_entry.get(), T.sep_or_tog)
                        print("\nTransforming channel 2")
                        ch2 = T.GPU_timepoints(T.vol[1,:,:,:], self.slices_entry.get(), T.sep_or_tog)
                        T.output = np.stack((ch1,ch2), axis=0)
                    else:
                        print("\nTransforming channel 1")
                        ch1 = T.GPU_timepoints(T.vol[:,0,:,:], self.slices_entry.get(), T.sep_or_tog)
                        print("\nTransforming channel 2")
                        ch2 = T.GPU_timepoints(T.vol[:,1,:,:], self.slices_entry.get(), T.sep_or_tog)
                        T.output = np.stack((ch1,ch2), axis=0)


                elif len(T.shape)==3:
                    print("\nTransforming channel 1")
                    ch1 = T.GPU_timepoints(T.vol[::2,:,:], self.slices_entry.get(), T.sep_or_tog)
                    print("\nTransforming channel 2")
                    ch2 = T.GPU_timepoints(T.vol[1::2,:,:], self.slices_entry.get(), T.sep_or_tog)
                    T.output = np.stack((ch1,ch2), axis=0)
            elif ch>2:
                print("Channels are more than 2, code must be updated")

            else:
                T.output = T.GPU_timepoints(T.vol, self.slices_entry.get(), T.sep_or_tog)

            print(f"Done transforming file {p}:")
            print(f"saving file...")
            tf.imwrite(os.path.join(recon_folder, f"recon_{p}.tif"), T.output)
            print("saving complete.")
        print(T.output.shape)
        print("Entire folder was been reconstructed succesfully")


#use classes

T = transformer(y_step_size, bin*pixelsize)

gui = gui_window()
