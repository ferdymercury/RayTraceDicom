/**
 * \file
 * \brief Header of helper functions and macros to check CUDA errors
 * \see https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */
#ifndef CUDA_ERRCHK_CUH
#define CUDA_ERRCHK_CUH

/**
 * \brief This function is used for checking GPU errors
 * \note It is a convenient macro that calls #cudaAssert adding file and line number automatically
 * \param ans ...
 */
#define cudaErrchk(ans) { cudaAssert((ans), __FILE__, __LINE__); }

/**
 * \brief GPU assert function
 * \param code ...
 * \param file ...
 * \param line ...
 * \param abort ...
 * \return void
 */
void cudaAssert(const cudaError_t code, char* const file, const int line, const bool abort=true);

#endif // CUDA_ERRCHK_CUH
