from panda3d.core import loadPrcFileData
#loadPrcFileData("", "window-type none" ) # Make sure we don't need a graphics engine (Will also prevent X errors / Display errors when starting on linux without X server)
loadPrcFileData("", "window-type offscreen" )
loadPrcFileData("", "audio-library-name null" ) # Prevent ALSA errors
loadPrcFileData('', 'show-frame-rate-meter true')
loadPrcFileData('', 'sync-video 0')
from direct.showbase.ShowBase import ShowBase
from panda3d.core import FrameBufferProperties, WindowProperties
from panda3d.core import GraphicsPipe, GraphicsOutput
from panda3d.core import Texture
from panda3d.core import AmbientLight, DirectionalLight, LightAttrib
from panda3d.core import NodePath
from panda3d.core import LVector3
from panda3d.core import Filename
from direct.interval.IntervalGlobal import *  # Needed to use Intervals
from direct.gui.DirectGui import *
from math import pi, sin
import sys
import os
import json
import torch
import numpy as np
from PIL import Image, ImageFilter
import folder_paths
import time
import cv2

comfy_path = os.path.dirname(folder_paths.__file__)
panda3d_models_path=f'{comfy_path}/custom_nodes/ComfyUI-Panda3d/models'

def show_rgbd_image(image, depth_image, window_name='Image window', delay=1, depth_offset=0.0, depth_scale=1.0):
    if depth_image.dtype != np.uint8:
        if depth_scale is None:
            depth_scale = depth_image.max() - depth_image.min()
        if depth_offset is None:
            depth_offset = depth_image.min()
        depth_image = np.clip((depth_image - depth_offset) / depth_scale, 0.0, 1.0)
        depth_image = (255.0 * depth_image).astype(np.uint8)
    depth_image = np.tile(depth_image, (1, 1, 3))
    if image.shape[2] == 4:  # add alpha channel
        alpha = np.full(depth_image.shape[:2] + (1,), 255, dtype=np.uint8)
        depth_image = np.concatenate([depth_image, alpha], axis=-1)
    images = np.concatenate([image, depth_image], axis=1)
    # images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)  # not needed since image is already in BGR format
    cv2.imshow(window_name, images)
    key = cv2.waitKey(delay)
    key &= 255
    if key == 27 or key == ord('q'):
        print("Pressed ESC or q, exiting")
        exit_request = True
    else:
        exit_request = False
    return exit_request

class Panda3dBase(ShowBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            }
        }

    RETURN_TYPES = ("Panda3dBase","Panda3dLoader","Panda3dModel",)
    RETURN_NAMES = ("base","loader","render",)
    FUNCTION = "run"
    CATEGORY = "Panda3d"

    def __init__(self):
        super(Panda3dBase, self).__init__()

        base.disableMouse()  # Allow manual positioning of the camera
        camera.setPosHpr(0, -8, 2.5, 0, -9, 0)

        #self.loadModels()  # Load and position our models
        #self.setupLights()  # Add some basic lighting
        #self.startCarousel()  # Create the needed intervals and put the
                              # carousel into motion

        # Needed for camera image
        self.dr = self.camNode.getDisplayRegion(0)

        # Needed for camera depth image
        winprops = WindowProperties.size(self.win.getXSize(), self.win.getYSize())
        fbprops = FrameBufferProperties()
        fbprops.setDepthBits(1)
        self.depthBuffer = self.graphicsEngine.makeOutput(
            self.pipe, "depth buffer", -2,
            fbprops, winprops,
            GraphicsPipe.BFRefuseWindow,
            self.win.getGsg(), self.win)
        self.depthTex = Texture()
        self.depthTex.setFormat(Texture.FDepthComponent)
        self.depthBuffer.addRenderTexture(self.depthTex,
            GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPDepth)
        lens = self.cam.node().getLens()
        # the near and far clipping distances can be changed if desired
        # lens.setNear(5.0)
        # lens.setFar(500.0)
        self.depthCam = self.makeCamera(self.depthBuffer,
            lens=lens,
            scene=render)
        self.depthCam.reparentTo(self.cam)

        # TODO: Scene is rendered twice: once for rgb and once for depth image.
        # How can both images be obtained in one rendering pass?
        
    def loadModels(self):
        # Load the carousel base
        self.carousel = loader.loadModel(Filename.fromOsSpecific(f"{panda3d_models_path}/carousel_base"))
        self.carousel.reparentTo(render)  # Attach it to render

        # Load the modeled lights that are on the outer rim of the carousel
        # (not Panda lights)
        # There are 2 groups of lights. At any given time, one group will have
        # the "on" texture and the other will have the "off" texture.
        self.lights1 = loader.loadModel(Filename.fromOsSpecific(f"{panda3d_models_path}/carousel_lights"))
        self.lights1.reparentTo(self.carousel)

        # Load the 2nd set of lights
        self.lights2 = loader.loadModel(Filename.fromOsSpecific(f"{panda3d_models_path}/carousel_lights"))
        # We need to rotate the 2nd so it doesn't overlap with the 1st set.
        self.lights2.setH(36)
        self.lights2.reparentTo(self.carousel)

        # Load the textures for the lights. One texture is for the "on" state,
        # the other is for the "off" state.
        self.lightOffTex = loader.loadTexture(Filename.fromOsSpecific(f"{panda3d_models_path}/carousel_lights_off.jpg"))
        self.lightOnTex = loader.loadTexture(Filename.fromOsSpecific(f"{panda3d_models_path}/carousel_lights_on.jpg"))

        # Create an list (self.pandas) with filled with 4 dummy nodes attached
        # to the carousel.
        # This uses a python concept called "Array Comprehensions."  Check the
        # Python manual for more information on how they work
        self.pandas = [self.carousel.attachNewNode("panda" + str(i))
                       for i in range(4)]
        self.models = [loader.loadModel(Filename.fromOsSpecific(f"{panda3d_models_path}/carousel_panda"))
                       for i in range(4)]
        self.moves = [0] * 4

        for i in range(4):
            # set the position and orientation of the ith panda node we just created
            # The Z value of the position will be the base height of the pandas.
            # The headings are multiplied by i to put each panda in its own position
            # around the carousel
            self.pandas[i].setPosHpr(0, 0, 1.3, i * 90, 0, 0)

            # Load the actual panda model, and parent it to its dummy node
            self.models[i].reparentTo(self.pandas[i])
            # Set the distance from the center. This distance is based on the way the
            # carousel was modeled in Maya
            self.models[i].setY(.85)

        # Load the environment (Sky sphere and ground plane)
        self.env = loader.loadModel(Filename.fromOsSpecific(f"{panda3d_models_path}/env"))
        self.env.reparentTo(render)
        self.env.setScale(7)

    # Panda Lighting
    def setupLights(self):
        # Create some lights and add them to the scene. By setting the lights on
        # render they affect the entire scene
        # Check out the lighting tutorial for more information on lights
        ambientLight = AmbientLight("ambientLight")
        ambientLight.setColor((.4, .4, .35, 1))
        directionalLight = DirectionalLight("directionalLight")
        directionalLight.setDirection(LVector3(0, 8, -2.5))
        directionalLight.setColor((0.9, 0.8, 0.9, 1))
        render.setLight(render.attachNewNode(directionalLight))
        render.setLight(render.attachNewNode(ambientLight))

        # Explicitly set the environment to not be lit
        self.env.setLightOff()

    def startCarousel(self):
        # Here's where we actually create the intervals to move the carousel
        # The first type of interval we use is one created directly from a NodePath
        # This interval tells the NodePath to vary its orientation (hpr) from its
        # current value (0,0,0) to (360,0,0) over 20 seconds. Intervals created from
        # NodePaths also exist for position, scale, color, and shear

        self.carouselSpin = self.carousel.hprInterval(20, LVector3(360, 0, 0))
        # Once an interval is created, we need to tell it to actually move.
        # start() will cause an interval to play once. loop() will tell an interval
        # to repeat once it finished. To keep the carousel turning, we use
        # loop()
        self.carouselSpin.loop()

        # The next type of interval we use is called a LerpFunc interval. It is
        # called that becuase it linearly interpolates (aka Lerp) values passed to
        # a function over a given amount of time.

        # In this specific case, horses on a carousel don't move contantly up,
        # suddenly stop, and then contantly move down again. Instead, they start
        # slowly, get fast in the middle, and slow down at the top. This motion is
        # close to a sine wave. This LerpFunc calls the function oscillatePanda
        # (which we will create below), which changes the height of the panda based
        # on the sin of the value passed in. In this way we achieve non-linear
        # motion by linearly changing the input to a function
        for i in range(4):
            self.moves[i] = LerpFunc(
                self.oscillatePanda,  # function to call
                duration=3,  # 3 second duration
                fromData=0,  # starting value (in radians)
                toData=2 * pi,  # ending value (2pi radians = 360 degrees)
                # Additional information to pass to
                # self.oscialtePanda
                extraArgs=[self.models[i], pi * (i % 2)]
            )
            # again, we want these to play continuously so we start them with
            # loop()
            self.moves[i].loop()

        # Finally, we combine Sequence, Parallel, Func, and Wait intervals,
        # to schedule texture swapping on the lights to simulate the lights turning
        # on and off.
        # Sequence intervals play other intervals in a sequence. In other words,
        # it waits for the current interval to finish before playing the next
        # one.
        # Parallel intervals play a group of intervals at the same time
        # Wait intervals simply do nothing for a given amount of time
        # Func intervals simply make a single function call. This is helpful because
        # it allows us to schedule functions to be called in a larger sequence. They
        # take virtually no time so they don't cause a Sequence to wait.

        self.lightBlink = Sequence(
            # For the first step in our sequence we will set the on texture on one
            # light and set the off texture on the other light at the same time
            Parallel(
                Func(self.lights1.setTexture, self.lightOnTex, 1),
                Func(self.lights2.setTexture, self.lightOffTex, 1)),
            Wait(1),  # Then we will wait 1 second
            # Then we will switch the textures at the same time
            Parallel(
                Func(self.lights1.setTexture, self.lightOffTex, 1),
                Func(self.lights2.setTexture, self.lightOnTex, 1)),
            Wait(1)  # Then we will wait another second
        )

        self.lightBlink.loop()  # Loop this sequence continuously

    def oscillatePanda(self, rad, panda, offset):
        # This is the oscillation function mentioned earlier. It takes in a
        # degree value, a NodePath to set the height on, and an offset. The
        # offset is there so that the different pandas can move opposite to
        # each other.  The .2 is the amplitude, so the height of the panda will
        # vary from -.2 to .2
        print(sin(rad + offset) * .2)
        panda.setZ(sin(rad + offset) * .2)

    def get_camera_image(self, requested_format=None):
        """
        Returns the camera's image, which is of type uint8 and has values
        between 0 and 255.
        The 'requested_format' argument should specify in which order the
        components of the image must be. For example, valid format strings are
        "RGBA" and "BGRA". By default, Panda's internal format "BGRA" is used,
        in which case no data is copied over.
        """
        tex = self.dr.getScreenshot()
        if requested_format is None:
            data = tex.getRamImage()
        else:
            data = tex.getRamImageAs(requested_format)
        image = np.frombuffer(data, np.uint8)  # use data.get_data() instead of data in python 2
        image.shape = (tex.getYSize(), tex.getXSize(), tex.getNumComponents())
        image = np.flipud(image)
        return image

    def get_camera_depth_image(self):
        """
        Returns the camera's depth image, which is of type float32 and has
        values between 0.0 and 1.0.
        """
        data = self.depthTex.getRamImage()
        depth_image = np.frombuffer(data, np.float32)
        depth_image.shape = (self.depthTex.getYSize(), self.depthTex.getXSize(), self.depthTex.getNumComponents())
        depth_image = np.flipud(depth_image)
        return depth_image

    def run(self):
        return (self,self.loader,self.render,)

class Panda3dLoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base":("Panda3dBase",),
                "loader":("Panda3dLoader",),
                "parent":("Panda3dModel",),
                "model_path": ("STRING",{"default":f"{panda3d_models_path}/carousel_base"}),
                "x": ("FLOAT", {"default": 0}),
                "y": ("FLOAT", {"default": 0}),
                "z": ("FLOAT", {"default": 0}),
                "h": ("FLOAT", {"default": 0}),
                "p": ("FLOAT", {"default": 0}),
                "r": ("FLOAT", {"default": 0}),
                "sx": ("FLOAT", {"default": 1}),
                "sy": ("FLOAT", {"default": 1}),
                "sz": ("FLOAT", {"default": 1}),
            }
        }
        
    RETURN_TYPES = ("Panda3dBase","Panda3dModel",)
    RETURN_NAMES = ("base","model",)
    FUNCTION = "run"
    CATEGORY = "Panda3d"

    def run(self,base,loader,parent,model_path,x,y,z,h,p,r,sx,sy,sz):
        model = loader.loadModel(Filename.fromOsSpecific(model_path))
        model.reparentTo(parent)
        model.setPos(x,y,z)
        model.setHpr(h,p,r)
        model.setScale(sx,sy,sz)
        return (base,model,)

class Panda3dTest:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base":("Panda3dBase",),
                "carousel":("Panda3dModel",),
                "frame_length": ("INT",{"default":14}),
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "Panda3d"

    def run(self,base,carousel,frame_length):
        outframes=[]
        #path = os.path.join(folder_paths.output_directory, file_name)
        #base = Panda3dBase()#windowType='offscreen'
        #base.run()
        radius = 20
        step = 1
        for i in range(frame_length):
            print(i)
            base.graphicsEngine.renderFrame()
            
            angleDegrees = i * step
            angleRadians = angleDegrees * (np.pi / 180.0)

            carousel.setHpr(angleDegrees, 0, 0)
            
            image = base.get_camera_image()
            #depth_image = base.get_camera_depth_image()
            #show_rgbd_image(image, depth_image)
            #base.screenshot(namePrefix='screenshot', defaultFilename=1, source=None, imageComment="")
            
            image_pil = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
            image_tensor_out = torch.tensor(np.array(image_pil).astype(np.float32) / 255.0)
            image_tensor_out = torch.unsqueeze(image_tensor_out, 0)
            outframes.append(image_tensor_out)
            Wait(1)
            
        #base.destroy()
        return torch.cat(tuple(outframes), dim=0).unsqueeze(0)

NODE_CLASS_MAPPINGS = {
    "Panda3dBase":Panda3dBase,
    "Panda3dLoadModel":Panda3dLoadModel,
    "Panda3dTest":Panda3dTest,
}

