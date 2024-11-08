/**
 * \file
 * \brief CUDA error assert function declarations
 * \see https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */
#ifndef CUDA_ERRCHK_CUH
#define CUDA_ERRCHK_CUH///< Header guard
#include "driver_types.h"

/**
 * \brief This function is used for checking GPU errors
 * \note It is a convenient macro that calls #cudaAssert adding file and line number automatically
 * \param ans the return code of the cuda function called
 */
#define cudaErrchk(ans) { cudaAssert((ans), __FILE__, __LINE__); }

/**
 * \brief GPU assert function
 * \param code the return code
 * \param file the filename where the macro was called
 * \param line the line in that file where the macro was called
 * \param abort if program should be aborted in case of errorcode found
 * \return void
 * \throw std::runtime_error if code!=cudaSuccess
 */
void cudaAssert(const cudaError_t code, const char* file, const int line, const bool abort=true);

#endif // CUDA_ERRCHK_CUH
