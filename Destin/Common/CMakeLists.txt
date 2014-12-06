include_directories(
    ../DavisDestin/include
    ../TreeMining
    ../TreeMining/CMOrderedTreeMiner
)

add_library( common SHARED
    DestinIterationFinishedCallback.h
    DestinNetworkAlt
    INetwork.h
    Transporter.h
    VideoSource
    VideoWriter
    unit_test.h
    CifarSource.h
    BeliefExporter.h
    ImageSourceBase.h
    ImageSourceImpl.h
    DestinTreeManager
    DestinTreeIteratorCallback
    CztMod.h
)

add_executable(testCifarSource TestCifar.cpp)
add_executable(testDestinNetworkAlt TestDestinNetworkAlt.cpp)
add_executable(openCogAtomGenerator OpenCogAtomGenerator.cpp)

find_package( OpenCV REQUIRED) 

# don't link avutil_lib with windows
if(WIN32 OR APPLE)
set(avutil_lib "")
else()
set(avutil_lib avutil)
endif()

target_link_libraries(testCifarSource ${OpenCV_LIBS})
target_link_libraries(common cm_ordered_treeminer_wrapper destinalt)
target_link_libraries(testDestinNetworkAlt common destinalt)
target_link_libraries(openCogAtomGenerator common destinalt ${avutil_lib})

install(TARGETS common testCifarSource testDestinNetworkAlt
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
)
