# GPU Architecture Detection Module
# Detects one supported architecture family from the locally installed GPUs.

function(detect_gpu_arch OUTPUT_VARIABLE)
    set(DETECTED_ARCH "")

    # AUTO must describe real installed GPUs, not architectures supported by
    # the compiler. Headless build hosts must use an explicit build target.
    find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi)
    if(NVIDIA_SMI_EXECUTABLE)
        execute_process(
            COMMAND ${NVIDIA_SMI_EXECUTABLE} --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE NVIDIA_SMI_OUTPUT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE NVIDIA_SMI_RESULT
        )
        if(NVIDIA_SMI_RESULT EQUAL 0 AND NVIDIA_SMI_OUTPUT)
            string(REPLACE "\n" ";" GPU_CAPABILITIES "${NVIDIA_SMI_OUTPUT}")
            set(DETECTED_ARCHES)
            foreach(COMPUTE_CAP IN LISTS GPU_CAPABILITIES)
                string(STRIP "${COMPUTE_CAP}" COMPUTE_CAP)
                string(REGEX MATCH "^([0-9]+)\\.([0-9]+)$" VALID_COMPUTE_CAP "${COMPUTE_CAP}")
                if(NOT VALID_COMPUTE_CAP)
                    continue()
                endif()

                string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)$" "\\1" CAP_MAJOR "${COMPUTE_CAP}")
                if(CAP_MAJOR EQUAL 8)
                    list(APPEND DETECTED_ARCHES "SM80")
                elseif(CAP_MAJOR EQUAL 9)
                    list(APPEND DETECTED_ARCHES "SM90")
                elseif(CAP_MAJOR GREATER_EQUAL 10)
                    list(APPEND DETECTED_ARCHES "SM100")
                endif()
            endforeach()

            list(REMOVE_DUPLICATES DETECTED_ARCHES)
            list(LENGTH DETECTED_ARCHES DETECTED_ARCH_COUNT)
            if(DETECTED_ARCH_COUNT EQUAL 1)
                list(GET DETECTED_ARCHES 0 DETECTED_ARCH)
                message(STATUS "Detected local GPU target: ${DETECTED_ARCH}")
            elseif(DETECTED_ARCH_COUNT GREATER 1)
                message(WARNING
                    "AUTO does not support heterogeneous GPU architecture families: ${DETECTED_ARCHES}"
                )
            endif()
        endif()
    endif()

    set(${OUTPUT_VARIABLE} ${DETECTED_ARCH} PARENT_SCOPE)
endfunction()
