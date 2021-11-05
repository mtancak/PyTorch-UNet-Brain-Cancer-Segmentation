# adapted from https://kitware.github.io/vtk-examples/site/Python/Medical/MedicalDemo4/ and https://kitware.github.io/vtk-examples/site/Python/Visualization/MultipleViewports/

import torch
import patchify
import os
import numpy as np

from vtkmodules.vtkCommonCore import VTK_UNSIGNED_SHORT
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersCore import (
    vtkFlyingEdges3D,
    vtkStripper)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCamera,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer)
import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2

from model import UNet3D


# https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
def crop_center(img,cropx,cropy,cropz):
    print("image shape = " + str(img.shape))
    z,y,x = img.shape[-3:]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    startz = z//2-(cropz//2)
    return img[:, :, startz:startz+cropz, starty:starty+cropy, startx:startx+cropx]


def get_volume(dir, fn, ext, model=None):
    dir_fns = os.listdir(dir)
    dir_fns = [x.replace(ext, '') for x in dir_fns if fn in x]
    print("dir_fns = " + str(dir_fns))

    patches_ind = np.array([x.split(fn)[1].split("_")[1:] for x in dir_fns], dtype=int)  # splits the grid coordinates in the names into a list of lists
    patches_dims = patches_ind.max(axis=0) + 1  # account for 0 indexing

    patch_shape = np.array(np.load(dir + dir_fns[0] + ext).shape)

    patches_shape = np.concatenate((patches_dims, patch_shape), axis=0)

    patches = np.zeros(patches_shape)

    ind_fn_pairs = zip(patches_ind, dir_fns)
    for i, f in ind_fn_pairs:
        patches[tuple(i)] = np.load(dir + f + ext)

    if len(patch_shape) % 2 == 0:
        if model:
            for z in range(patches.shape[0]):
                for y in range(patches.shape[1]):
                    for x in range(patches.shape[2]):
                        patches[z][y][x] = model.forward(
                            torch.tensor(patches[z][y][x]).unsqueeze(0).cuda().float()).detach().cpu().numpy()
        patches = np.moveaxis(patches, 3, 0)
        vol = []
        for c in patches:
            vol.append(patchify.unpatchify(c, tuple(patch_shape[1:] * patches_dims)))
        vol = np.stack(vol)
        if model:
            vol = np.argmax(vol, axis=0)
        else:
            vol = vol[0]
    else:
        vol = patchify.unpatchify(patches, tuple(patch_shape * patches_dims))

    vol_image = vtkImageData()
    vol_image.SetDimensions(vol.shape)
    vol_image.SetSpacing(1.0, 1.0, 1.0)
    vol_image.AllocateScalars(VTK_UNSIGNED_SHORT, 1)

    for z in range(vol.shape[0]):
        for y in range(vol.shape[1]):
            for x in range(vol.shape[2]):
                vol_image.SetScalarComponentFromDouble(z, y, x, 0, vol[z][y][x])

    return vol_image


colors = vtkNamedColors()

colors.SetColor('SkinColor', [240, 184, 160, 50])
colors.SetColor('seg1c', [0, 0, 255, 50])
colors.SetColor('seg2c', [0, 255, 0, 50])
colors.SetColor('seg3c', [255, 0, 0, 50])
colors.SetColor('BkgColor', [80, 80, 80, 255])

def get_actor(image, clr, opacity, threshold):
    # An isosurface, or contour value of 1150 is known to correspond to the
    # bone of the patient.
    # The triangle stripper is used to create triangle strips from the
    # isosurface these render much faster on may systems.
    extractor = vtkFlyingEdges3D()
    extractor.SetInputData(image)
    extractor.SetValue(0, threshold)

    stripper = vtkStripper()
    stripper.SetInputConnection(extractor.GetOutputPort())

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(stripper.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetDiffuseColor(colors.GetColor3d(clr))
    actor.GetProperty().SetSpecular(0.3)
    actor.GetProperty().SetSpecularPower(20)
    actor.GetProperty().SetOpacity(opacity)

    return actor


def main():
    model = UNet3D(in_channels=4, out_channels=4).to("cuda")
    model.load_state_dict(torch.load("./model_10"))

    dir = "C:/Users/Milan/Documents/Fast_Datasets/BraTS20-Long/prep/val/"
    fn = "BraTS20_Training_004"
    ext = ".npy"

    data_vol_image = get_volume(dir + "data/", fn, ext)
    pred_vol_image = get_volume(dir + "data/", fn, ext, model=model)
    mask_vol_image = get_volume(dir + "mask/", fn, ext)

    skin_actor = get_actor(data_vol_image, 'SkinColor', 0.1, 10)
    edema_actor = get_actor(pred_vol_image, 'seg1c', 0.2, 1)
    non_enhancing_actor = get_actor(pred_vol_image, 'seg2c', 0.3, 2)
    gd_enhancing_actor = get_actor(pred_vol_image, 'seg3c', 1.0, 3)

    # Create the renderer, the render window, and the interactor. The renderer
    # draws into the render window, the interactor enables mouse- and
    # keyboard-based interaction with the data within the render window.
    a_renderer = vtkRenderer()
    ren_win = vtkRenderWindow()
    ren_win.AddRenderer(a_renderer)
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)

    # It is convenient to create an initial view of the data. The FocalPoint
    # and Position form a vector direction. Later on (ResetCamera() method)
    # this vector is used to position the camera to look at the data in
    # this direction.
    a_camera = vtkCamera()
    a_camera.SetViewUp(0, 0, -1)
    a_camera.SetPosition(0, -1, 0)
    a_camera.SetFocalPoint(0, 0, 0)
    a_camera.ComputeViewPlaneNormal()
    a_camera.Azimuth(30.0)
    a_camera.Elevation(30.0)

    # Actors are added to the renderer. An initial camera view is created.
    # The Dolly() method moves the camera towards the FocalPoint,
    # thereby enlarging the image.
    a_renderer.AddActor(skin_actor)
    a_renderer.AddActor(edema_actor)
    a_renderer.AddActor(non_enhancing_actor)
    a_renderer.AddActor(gd_enhancing_actor)
    a_renderer.SetActiveCamera(a_camera)
    a_renderer.ResetCamera()
    a_camera.Dolly(1.5)

    # Set a background color for the renderer and set the size of the
    # render window (expressed in pixels).
    a_renderer.SetBackground(colors.GetColor3d('BkgColor'))
    ren_win.SetSize(640, 480)

    # Note that when camera movement occurs (as it does in the Dolly()
    # method), the clipping planes often need adjusting. Clipping planes
    # consist of two planes: near and far along the view direction. The
    # near plane clips out objects in front of the plane the far plane
    # clips out objects behind the plane. This way only what is drawn
    # between the planes is actually rendered.
    a_renderer.ResetCameraClippingRange()

    # Initialize the event loop and then start it.
    iren.Initialize()
    iren.Start()

if __name__ == '__main__':
    main()