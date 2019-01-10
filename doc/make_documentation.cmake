#From https://github.com/ejerome/CMakeUtilities/blob/master/targets/make_documentation.cmake
#http://jesnault.fr/website/document-your-cmake-code-within-doxygen/

## pragma once
if(_MAKE_DOCUMENTATION_CMAKE_INCLUDED_)
  return()
endif()
set(_MAKE_DOCUMENTATION_CMAKE_INCLUDED_ true)

cmake_minimum_required(VERSION 2.8)

include(${CMAKE_ROOT}/Modules/CMakeParseArguments.cmake)

## HOW TO DOCUMENT YOUR CMAKE FILES :
## 1- Use CMAKE_DOCUMENTATION_START <name> flag to start your documentation
## 2- Document your cmake file like you want (you can use doxygen syntax)
## 3- Use CMAKE_DOCUMENTATION_END flag to finish your documentation
##
## The parser will generate a documentation section foreach filenames parsed with a section foreach macros/functions under the file parsed.
## <name> could be :
##  * a fileName :
##      In this case, the parser will add the doxygen documentation of the filename.
##  * a simple name (function/macro) name :
##      In this case, the parser will add a section under the filename one.
##
## Example:
##    ## <CMAKE_DOC_START_FLAG> FindMySDK.cmake
##    ##
##    ## Try to find MySDK.
##    ## Once done this will define :
##    ##  \\li MYSDK_FOUND - system has MYSDK
##    ##  \\li MYSDK_INCLUDE_DIR - the MYSDK include directory
##    ##  \\li MYSDK_LIBRARIES - The libraries provided by MYSDK
##    ##
##    ## You can use the MYSDK_DIR environment variable
##    ##
##    ## <CMAKE_DOC_END_FLAG>
##
## This file is used to generated cmake documentation
## It allow to create cmakedoc custom_target (see docs/CMakeLists.txt)~
##
## In your CMakeLists.txt, just include this file.
## Or call subsequently :
##  PARSE_CMAKE_DOCUMENTATION()
##  WRITE_CMAKE_DOCUMENTATION( "${CMAKE_SOURCE_DIR}/cmake.dox" SORTED )
##
## Then the standard Doxygen application will be able to parse your cmake.dox documentation to be include.

## SEE THE REAL COMMAND AT THE END OF THIS FILE !

## ###########################################################################
## #####################   INTERNAL USE   ####################################
## ###########################################################################

## Parse for a list of cmake files in order to recover documentation notes.
##
## PARSE_CMAKE_DOCUMENTATION(
##      [INCLUDES   includeFilePathPattern] ## default ${CMAKE_SOURCE_DIR}/*.cmake*
##      [EXCLUDES   excludeFilePathPattern] ## default this file (avoiding conflicts)
##      [START_FLAG matchingStartFlag]      ## CMAKE_DOCUMENTATION_START
##      [END_FLAG   matchingEndFlag]        ## CMAKE_DOCUMENTATION_END
##)
##
MACRO(PARSE_CMAKE_DOCUMENTATION )

    ##params: "PREFIX" "optionsArgs" "oneValueArgs" "multiValueArgs"
    cmake_parse_arguments(PARSE_CMAKE_DOCUMENTATION "" "START_FLAG;END_FLAG" "INCLUDES;EXCLUDES" ${ARGN} )

    # INCLUDES cmake file to the list files
    if(NOT DEFINED PARSE_CMAKE_DOCUMENTATION_INCLUDES)
        set(PARSE_CMAKE_DOCUMENTATION_INCLUDES "${CMAKE_SOURCE_DIR}/../CMakeLists.txt") # all *.cmake* by default
    endif()
    message(cmake_files_list)
    foreach(includeFilePathPattern ${PARSE_CMAKE_DOCUMENTATION_INCLUDES})
        file(GLOB_RECURSE cmake_files "${includeFilePathPattern}")
        list(APPEND cmake_files_list ${cmake_files})
    endforeach()

    # EXCLUDES cmake file to the list files
    if(NOT DEFINED PARSE_CMAKE_DOCUMENTATION_EXCLUDES)
        set(PARSE_CMAKE_DOCUMENTATION_EXCLUDES
                "${CMAKE_SOURCE_DIR}/*make_documentation.cmake*") # mutual exclude by default
    endif()
    foreach(excludeFilePathPattern ${PARSE_CMAKE_DOCUMENTATION_EXCLUDES})
        file(GLOB_RECURSE cmake_files_exclude "${excludeFilePathPattern}")
        list(REMOVE_ITEM cmake_files_list ${cmake_files_exclude})
        #message("remove file from cmake documentation files list : ${cmake_files_exclude}")
    endforeach()

    # default START_FLAG
    if(NOT DEFINED PARSE_CMAKE_DOCUMENTATION_START_FLAG)
        set(PARSE_CMAKE_DOCUMENTATION_START_FLAG "CMAKE_DOCUMENTATION_START")
    endif()

    # default END_FLAG
    if(NOT DEFINED PARSE_CMAKE_DOCUMENTATION_END_FLAG)
        set(PARSE_CMAKE_DOCUMENTATION_END_FLAG "CMAKE_DOCUMENTATION_END")
    endif()

    # Process for each file of the list
    foreach(cmake_file ${cmake_files_list})
        file(READ ${cmake_file} cmake_file_content)
        message(STATUS "Generate cmake doc for : ${cmake_file}")

        ## check pair tags for cmake doc
        string(REGEX MATCHALL "${PARSE_CMAKE_DOCUMENTATION_START_FLAG}" matches_start ${cmake_file_content})
        list(LENGTH matches_start matches_start_count)
        string(REGEX MATCHALL "${PARSE_CMAKE_DOCUMENTATION_END_FLAG}" matches_end ${cmake_file_content})
        list(LENGTH matches_end matches_end_count)
        if(NOT matches_start_count EQUAL matches_end_count)
            message("WARNING: You forgot a tag for cmake documentation generation.
                    Matches ${PARSE_CMAKE_DOCUMENTATION_START_FLAG} = ${matches_start_count}.
                    Matches ${PARSE_CMAKE_DOCUMENTATION_END_FLAG} = ${matches_end_count}.
                    The cmake file ${cmake_file} will not be parse for cmake doc generation.")
            set(cmake_file_content "") # to skip the parsing
        endif()

        ## Parse the cmake file
        string(REGEX REPLACE "\r?\n" ";" cmake_file_content "${cmake_file_content}")
        set(sectionStarted false)
        set(docSection "")
        set(docSectionContent "")
        foreach(line ${cmake_file_content})

            ## find the start balise
            string(REGEX MATCH "##.*${PARSE_CMAKE_DOCUMENTATION_START_FLAG}.*" matchStart ${line})
            if( matchStart AND NOT sectionStarted)
                set(sectionStarted true)
                string(REGEX REPLACE "##.*${PARSE_CMAKE_DOCUMENTATION_START_FLAG}(.*)" "\\1" docSection ${matchStart})
            endif()

            ## find the end balise
            string(REGEX MATCH "##.*${PARSE_CMAKE_DOCUMENTATION_END_FLAG}" matchEnd ${line})
            if( matchEnd AND sectionStarted)
                string(REGEX REPLACE ";" "" docSectionContent "${docSectionContent}") ## to remove ; at each line
                if(NOT docSection)
                    set(docSection "other")
                endif()
                ###########################
                process_cmake_documentation( "${cmake_file}" "${docSection}" "${docSectionContent}" )
                ###########################
                set(sectionStarted false)
                set(docSection "")
                set(ligneComment "")
                set(docSectionContent "")
            endif()

            ## Extract comment between balises
            string(REGEX MATCH "#.*" matchComment ${line})
            if( matchComment AND sectionStarted AND NOT matchStart AND NOT matchEnd)
                string(REGEX REPLACE "#" "" ligneComment ${matchComment})
                list(APPEND docSectionContent ${ligneComment})
            endif()

        endforeach()

    endforeach()
ENDMACRO()


## This macro define some global cmake variables to be used when generate the .dox file
## Internal used only
MACRO(PROCESS_CMAKE_DOCUMENTATION cmakeFile sectionName sectionContent)

    ## Strip cmakeFile path
    string(REGEX REPLACE "${CMAKE_SOURCE_DIR}" "" cmakeFileStrip ${cmakeFile})

    ## If sectionName is a fileName, add it to the list a file with its description
    string(REGEX MATCH ".*[.].*" fileExtensionsExist ${sectionName} )
    if(fileExtensionsExist)  # for .cmake
        string(REGEX REPLACE "(.*)[.].*" "\\1" sectionFileName ${fileExtensionsExist} )

        string(REGEX MATCH ".*[.].*" fileExtensionExist ${sectionFileName} )
        if(fileExtensionExist)  # double extension match (for .in)
            string(REGEX REPLACE "(.*)[.].*" "\\1" sectionFileName ${fileExtensionExist} )
        endif()

        ## section file name list
        list(APPEND gPROCESS_CMAKE_DOCUMENTATION_FILES_LIST "\\li \\ref ${sectionFileName}")
        ## section file description list
        list(APPEND gPROCESS_CMAKE_DOCUMENTATION_FILES_DESCRIPTIONS
            "
            \\subsection ${sectionFileName} ${sectionFileName}
            \\par
            \\e From: ${cmakeFileStrip} \\n
            ${sectionContent}
            "
        )
    else()
        ## section doc name list
        list(APPEND gPROCESS_CMAKE_DOCUMENTATION_SECTION_LIST "\\li \\ref ${sectionName}")
        ## section doc description list
        list(APPEND gPROCESS_CMAKE_DOCUMENTATION_SECTION_CONTENT
                "
                \\subsection ${sectionName} ${sectionName}
                \\par
                \\e From: ${cmakeFileStrip} \\n
                ${sectionContent}
                "
        )
    endif()

ENDMACRO()


## Generate the .dox file using globale variables set by previous macro
## WRITE_CMAKE_DOCUMENTATION( <outputPathFileName>
##      [SORTED]
##      [HEADER myCustomHeaderString]
##      [FOOTER myCustomFooterString]
## )
FUNCTION(WRITE_CMAKE_DOCUMENTATION outputPathFileName)

    ##params: "PREFIX" "optionsArgs" "oneValueArgs" "multiValueArgs"
    cmake_parse_arguments(WRITE_CMAKE_DOCUMENTATION "SORTED" "HEADER;FOOTER" "" ${ARGN} )

    file(WRITE  "${outputPathFileName}" "/*! \\page cmakePage CMake documentation")

    ## header file
    list(APPEND header "${WRITE_CMAKE_DOCUMENTATION_HEADER}

        MASTER INDEX CMAKE TOOLS

        \\li \\ref cmake_doc
        \\li \\ref cmake_files
    ")
    file(APPEND  "${outputPathFileName}" "${header}")


    ## SORT global lists
    if(WRITE_CMAKE_DOCUMENTATION_SORTED)
        if(gPROCESS_CMAKE_DOCUMENTATION_FILES_LIST)
            list(SORT   gPROCESS_CMAKE_DOCUMENTATION_FILES_LIST)
        endif()
        if(gPROCESS_CMAKE_DOCUMENTATION_FILES_DESCRIPTIONS)
            list(SORT   gPROCESS_CMAKE_DOCUMENTATION_FILES_DESCRIPTIONS)
        endif()
        if(gPROCESS_CMAKE_DOCUMENTATION_SECTION_LIST)
            list(SORT   gPROCESS_CMAKE_DOCUMENTATION_SECTION_LIST)
        endif()
        if(gPROCESS_CMAKE_DOCUMENTATION_SECTION_CONTENT)
            list(SORT   gPROCESS_CMAKE_DOCUMENTATION_SECTION_CONTENT)
        endif()
    endif()

    ## STRIP global lists
    if(gPROCESS_CMAKE_DOCUMENTATION_FILES_LIST)
        list(REMOVE_DUPLICATES  gPROCESS_CMAKE_DOCUMENTATION_FILES_LIST)
    endif()
    string(REGEX REPLACE ";" ""
        gPROCESS_CMAKE_DOCUMENTATION_FILES_LIST
        "${gPROCESS_CMAKE_DOCUMENTATION_FILES_LIST}"
    )
    string(REGEX REPLACE ";" ""
            gPROCESS_CMAKE_DOCUMENTATION_SECTION_LIST
            "${gPROCESS_CMAKE_DOCUMENTATION_SECTION_LIST}"
    )

    ## Fill content documentation
    set(content
        "
        \\section cmake_doc CMAKE DOC
        ${gPROCESS_CMAKE_DOCUMENTATION_SECTION_LIST}
        ${gPROCESS_CMAKE_DOCUMENTATION_SECTION_CONTENT}

        \\section cmake_files CMAKE FILES
        ${gPROCESS_CMAKE_DOCUMENTATION_FILES_LIST}
        ${gPROCESS_CMAKE_DOCUMENTATION_FILES_DESCRIPTIONS}
        "
    )
    file(APPEND  "${outputPathFileName}" "${content}")


    ## footer file
    set(footer "${WRITE_CMAKE_DOCUMENTATION_FOOTER} \n*/")
    file(APPEND "${outputPathFileName}" "${footer}")

ENDFUNCTION()


## ###########################################################################
## #####################   WHAT WE DO   ######################################
## ###########################################################################

PARSE_CMAKE_DOCUMENTATION()
WRITE_CMAKE_DOCUMENTATION( "${CMAKE_SOURCE_DIR}/cmake.dox" SORTED )


