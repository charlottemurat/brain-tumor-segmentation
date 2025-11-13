cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(BrainTumorSegmentation)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/BrainTumorSegmentation")
set(EXTENSION_CONTRIBUTORS "Sam Bard, Charlotte Murat, Jack Walker")
set(EXTENSION_DESCRIPTION "Module for automatic segmentation of brain tumors")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/BrainTumorSegmentation.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/BrainTumorSegmentation/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(BrainTumorSegmentation)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
