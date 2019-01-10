/**
 * \file
 * \brief CUDA error assert function declarations
 * \see https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */
#ifndef CUDA_ERRCHK_CUH
#define CUDA_ERRCHK_CUH///< Header guard

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
 * \throw std::runtime_error if code!=cudaSuccess
 */
void cudaAssert(const cudaError_t code, char* const file, const int line, const bool abort=true);

#endif // CUDA_ERRCHK_CUH
