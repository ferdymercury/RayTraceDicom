/**
 * \file
 * \brief Config runtime parameters header
 */
#ifndef CONFIG_H
#define CONFIG_H///< Header guard

#include <string>

/**
 * \brief Config runtime parameters
 */
struct Config
{
    /**
     * \brief Constructor
     * \param argc number of arguments passed through command line
     * \param argv pointer to string arguments
     * \see main(int, char**) function
     */
    Config(int argc, char **argv);

    unsigned short gpu_id{0};       ///< GPU id
    std::string ct_dir;             ///< Directory where the DICOM image series are stored
    std::string rtplan;             ///< RT plan filename
    std::string output_directory;   ///< Directory where the calculated results will be stored

    bool exit{false};               ///< Constructor exited because of error
};

#endif // CONFIG_H
