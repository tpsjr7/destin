FIND_PACKAGE(PythonLibs)
    
IF(PYTHONLIBS_FOUND)
    SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}") # so I can use the custom FindNumpy.cmake
    FIND_PACKAGE(NumPy)

    INCLUDE_DIRECTORIES( ${PYTHON_INCLUDE_DIRS} ${PYTHON_NUMPY_INCLUDE_DIR})
    SET_SOURCE_FILES_PROPERTIES(swig_destin_python.i PROPERTIES CPLUSPLUS ON)
        
    SET(CMAKE_SWIG_OUTDIR . )            
    SWIG_ADD_MODULE(pydestin python swig_destin_python.i)
    
    SWIG_LINK_LIBRARIES(pydestin ${OpenCV_LIBS} destinalt common som cm_ordered_treeminer_wrapper ${PYTHON_LIBRARY})

    IF(WIN32)
        SET(PY_LIB_DEST bin) # we want the generated _pydestin.pyd to be in bin
        INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/pydestin.py DESTINATION bin)
    ELSE()
        SET(PY_LIB_DEST lib)
    ENDIF()

    INSTALL(TARGETS _pydestin
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION ${PY_LIB_DEST}
        ARCHIVE DESTINATION lib/static
    )

ENDIF(PYTHONLIBS_FOUND)
