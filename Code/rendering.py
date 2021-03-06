# adapted from https://kitware.github.io/vtk-examples/site/Python/Medical/MedicalDemo4/
# and https://kitware.github.io/vtk-examples/site/Python/Visualization/MultipleViewports/

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
    vtkRenderer,
    vtkTextActor,
    vtkTextMapper,
    vtkTextProperty)
import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2
import vtkmodules.vtkRenderingFreeType

from model import UNet3D
from load_hyperparameters import hp


def get_volume(directory, fn, ext, model=None):
    dir_fns = os.listdir(directory)
    dir_fns = [x.replace(ext, '') for x in dir_fns if fn in x]

    # split the grid coordinates in the names into a list of lists
    patches_ind = np.array([x.split(fn)[1].split("_")[1:] for x in dir_fns], dtype=int)
    # get the highest indices in each dimension, giving us the patch grid shape
    patches_dims = patches_ind.max(axis=0) + 1  # account for 0 indexing

    # get the shape of the first entry, assume the rest are the same
    patch_shape = np.array(np.load(directory + dir_fns[0] + ext).shape)

    # concatenate the two arrays, this will be the shape of our grid
    patches_shape = np.concatenate((patches_dims, patch_shape), axis=0)

    patches = np.zeros(patches_shape)

    # load the data
    ind_fn_pairs = zip(patches_ind, dir_fns)
    for i, f in ind_fn_pairs:
        patches[tuple(i)] = np.load(directory + f + ext)

    # if true, we loaded in data, not a mask
    if len(patch_shape) % 2 == 0:
        # given a model, make predictions
        if model:
            for z in range(patches.shape[0]):
                for y in range(patches.shape[1]):
                    for x in range(patches.shape[2]):
                        patches[z][y][x] = model.forward(
                            torch.tensor(patches[z][y][x]).unsqueeze(0).cuda().float()).detach().cpu().numpy()
        patches = np.moveaxis(patches, 3, 0)
        # unpatchify the data
        vol = []
        for c in patches:
            vol.append(patchify.unpatchify(c, tuple(patch_shape[1:] * patches_dims)))
        vol = np.stack(vol)
        if model:
            vol = np.argmax(vol, axis=0)
        else:
            vol = vol[0]
    else:  # else we loaded in a mask, so simply unpatchify it
        vol = patchify.unpatchify(patches, tuple(patch_shape * patches_dims))

    vol_image = vtkImageData()
    vol_image.SetDimensions(vol.shape)
    vol_image.SetSpacing(1.0, 1.0, 1.0)
    vol_image.AllocateScalars(VTK_UNSIGNED_SHORT, 1)

    # inserts data into our vtkImageData volume
    for z in range(vol.shape[0]):
        for y in range(vol.shape[1]):
            for x in range(vol.shape[2]):
                vol_image.SetScalarComponentFromDouble(z, y, x, 0, vol[z][y][x])

    return vol_image


colors = vtkNamedColors()

colors.SetColor('SkinColor', [240, 184, 160, 50])
colors.SetColor('seg1c', [0, 0, 255, 50])  # peritumoral edema
colors.SetColor('seg2c', [0, 255, 0, 50])  # non-enhancing tumor core
colors.SetColor('seg3c', [255, 0, 0, 50])  # GD-enhancing tumor
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

    dir = hp["validation_dir"]
    fn = hp["rendering_sample_name"]
    ext = ".npy"

    data_vol_image = get_volume(dir + hp["data_dir_name"], fn, ext)
    pred_vol_image = get_volume(dir + hp["data_dir_name"], fn, ext, model=model)
    mask_vol_image = get_volume(dir + hp["seg_dir_name"], fn, ext)

    skin_actor = get_actor(data_vol_image, 'SkinColor', 0.1, 10)

    pred_edema_actor = get_actor(pred_vol_image, 'seg1c', 0.2, 1)
    pred_non_enhancing_actor = get_actor(pred_vol_image, 'seg2c', 0.3, 2)
    pred_gd_enhancing_actor = get_actor(pred_vol_image, 'seg3c', 1.0, 3)
    pred_actors = [pred_edema_actor, pred_non_enhancing_actor, pred_gd_enhancing_actor]

    mask_edema_actor = get_actor(mask_vol_image, 'seg1c', 0.2, 1)
    mask_non_enhancing_actor = get_actor(mask_vol_image, 'seg2c', 0.3, 2)
    mask_gd_enhancing_actor = get_actor(mask_vol_image, 'seg3c', 1.0, 3)
    mask_actors = [mask_edema_actor, mask_non_enhancing_actor, mask_gd_enhancing_actor]

    # One render window, multiple viewports.
    rw = vtkRenderWindow()
    rw.SetSize(1200, 600)

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(rw)

    # Define viewport ranges.
    xmins = [0, .5]
    xmaxs = [0.5, 1]
    ymins = [0, 0]
    ymaxs = [1, 1]

    # Share a camera between viewports
    camera = vtkCamera()

    for i, (actors, name) in enumerate([tuple((mask_actors, "Target")), tuple((pred_actors, "Prediction"))]):
        ren = vtkRenderer()
        rw.AddRenderer(ren)
        ren.SetViewport(xmins[i], ymins[i], xmaxs[i], ymaxs[i])

        ren.SetActiveCamera(camera)

        # Create a mapper and act
        ren.AddActor(skin_actor)
        ren.AddActor(actors[0])
        ren.AddActor(actors[1])
        ren.AddActor(actors[2])

        txt = vtkTextActor()
        txt.SetInput(name)
        rw_w, rw_h = rw.GetSize()
        txt.SetDisplayPosition(int(((rw_w * i) + (rw_w / 2.0)) / 2.0), int(rw_h * 0.1))
        ren.AddActor(txt)

        ren.ResetCamera()

    rw.Render()
    rw.SetWindowName('Milan Tancak')
    iren.Start()


if __name__ == '__main__':
    main()
