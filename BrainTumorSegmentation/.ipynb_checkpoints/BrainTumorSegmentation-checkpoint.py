import logging
import os
from typing import Annotated, Optional

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode
import SimpleITK as sitk

import numpy
from scipy.ndimage import gaussian_filter, convolve

#
# BrainTumorSegmentation
#


class BrainTumorSegmentation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("BrainTumorSegmentation")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#BrainTumorSegmentation">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # BrainTumorSegmentation1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="BrainTumorSegmentation",
        sampleName="BrainTumorSegmentation1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "BrainTumorSegmentation1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="BrainTumorSegmentation1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="BrainTumorSegmentation1",
    )

    # BrainTumorSegmentation2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="BrainTumorSegmentation",
        sampleName="BrainTumorSegmentation2",
        thumbnailFileName=os.path.join(iconsPath, "BrainTumorSegmentation2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="BrainTumorSegmentation2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="BrainTumorSegmentation2",
    )


#
# BrainTumorSegmentationParameterNode
#


@parameterNodeWrapper
class BrainTumorSegmentationParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    processedVolume - The output volume that have been processed and contain the segmentation
    """

    inputVolume: vtkMRMLScalarVolumeNode
    processedVolume: vtkMRMLScalarVolumeNode


#
# BrainTumorSegmentationWidget
#


class BrainTumorSegmentationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)
        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.layout.addWidget(self.inputSelector)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/BrainTumorSegmentation.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = BrainTumorSegmentationLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[BrainTumorSegmentationParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.processedVolume:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output volume nodes")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            sigma = self.ui.blurStrengthSlider.value
            alpha = self.ui.sharpStrengthSlider.value
            lower = self.ui.lowerThreshold.value
            upper = self.ui.upperThreshold.value
            inputVolume = self.inputSelector.currentNode()
            outputVolume = self.ui.outputSelector.currentNode()

            showInput = self.ui.showInput.checked
            showMask = self.ui.showMask.checked
            
            self.logic.process(inputVolume, outputVolume, sigma=sigma, alpha=alpha, lower=lower, upper=upper)
            vrLogic = slicer.modules.volumerendering.logic()
    
            maskNode = vrLogic.CreateDefaultVolumeRenderingNodes(outputVolume)
            maskNode.SetVisibility(showMask)
    
            maskPropertyNode = maskNode.GetVolumePropertyNode() #setting green color and full opaqueness for mask
            maskProperty = maskPropertyNode.GetVolumeProperty()
            colorTransferFunction = maskProperty.GetRGBTransferFunction(0)
            colorTransferFunction.RemoveAllPoints()
            colorTransferFunction.AddRGBPoint(1, 0.0, 1.0, 0.0)
            maskOpacityFunction = maskProperty.GetScalarOpacity()
            maskOpacityFunction.RemoveAllPoints()
            maskOpacityFunction.AddPoint(0, 0.0)
            maskOpacityFunction.AddPoint(1, 1.0)
    
            inputNode = vrLogic.CreateDefaultVolumeRenderingNodes(inputVolume) #setting transparency for input volume overlay
            inputNode.SetVisibility(showInput)
    
            inputPropertyNode = inputNode.GetVolumePropertyNode()
            inputProperty = inputPropertyNode.GetVolumeProperty()
            inputOpacityFunction = inputProperty.GetScalarOpacity()
            inputOpacityFunction.RemoveAllPoints()
            scalarRange = inputVolume.GetImageData().GetScalarRange()
            inputOpacityFunction.AddPoint(scalarRange[0], 0.0)
            inputOpacityFunction.AddPoint(scalarRange[1], 0.2) #20% opacity
    
            slicer.app.processEvents()  #refresh render

#
# BrainTumorSegmentationLogic
#


class BrainTumorSegmentationLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return BrainTumorSegmentationParameterNode(super().getParameterNode())

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                sigma = 1.0, alpha = 1.0, lower = 0.25, upper = 0.75,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param showResult: show output volume in slice viewers
        """
        
        if not outputVolume.GetDisplayNode(): #create a default output display node if one doesn't exist
            outputVolume.CreateDefaultDisplayNodes()

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")

        outputVolume.SetOrigin(inputVolume.GetOrigin())
        outputVolume.SetSpacing(inputVolume.GetSpacing())

        ijk_to_RAS = vtk.vtkMatrix4x4()
        inputVolume.GetIJKToRASMatrix(ijk_to_RAS)
        outputVolume.SetIJKToRASMatrix(ijk_to_RAS)
        #at this point output volume has all same spacing, origin, etc... as input
        #all filtering/manipulation should come after this

        
        volume_numpy = slicer.util.arrayFromVolume(inputVolume)
        sitk_image = sitk.GetImageFromArray(volume_numpy) #sitk takes Z, Y, X array order
        blur_image = sitk.SmoothingRecursiveGaussian(sitk_image, sigma=sigma)

        high_freq = sitk.Subtract(sitk_image, blur_image)
        sharpened = sitk.Add(sitk_image, sitk.Multiply(high_freq, alpha))
        
        filtered_volume = sitk.GetArrayFromImage(sharpened)
        
        slicer.util.updateVolumeFromArray(outputVolume, filtered_volume)
        displayNode = outputVolume.GetDisplayNode()
        if displayNode:
            scalarRange = outputVolume.GetImageData().GetScalarRange()
            displayNode.SetWindow(scalarRange[1] - scalarRange[0])
            displayNode.SetLevel(0.5 * (scalarRange[0] + scalarRange[1]))


        # Thresholding
        # arr = slicer.util.arrayFromVolume(inputVolume) 

        # sitkImage = sitk.GetImageFromArray(arr)
        # sitkImage.SetSpacing(inputVolume.GetSpacing())

        sitkImage = sharpened
        arr = filtered_volume

        maxValue = arr.max()
        # print('\n')
        # print(maxValue)
        # print(lower*maxValue)
        # print(upper*maxValue)
        
        binaryMask = sitk.BinaryThreshold(sitkImage, lowerThreshold=lower*maxValue, upperThreshold=upper*maxValue, insideValue=1, outsideValue=0)
        # binaryMask = sitk.BinaryThreshold(sitkImage, lowerThreshold=thresholdValue, upperThreshold=1e9, insideValue=1, outsideValue=0)
        
        cc = sitk.ConnectedComponent(binaryMask)
        largest = sitk.RelabelComponent(cc, sortByObjectSize=True)
        largestMask = largest == 1

        maskedImage = sitk.Mask(sitkImage, largestMask)

        maskedArr = sitk.GetArrayFromImage(maskedImage)
        slicer.util.updateVolumeFromArray(outputVolume, maskedArr)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")


#
# BrainTumorSegmentationTest
#


class BrainTumorSegmentationTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_BrainTumorSegmentation1()

    def test_BrainTumorSegmentation1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("BrainTumorSegmentation1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
 

        # Test the module logic

        logic = BrainTumorSegmentationLogic()

        # Test algorithm
        logic.process(inputVolume, outputVolume)
        
        
        
        self.delayDisplay("Test passed")
