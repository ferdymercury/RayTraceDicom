/**
 * \file
 * \brief Config runtime parameters implementation
 */

#include "config.h"
#include "CLI/CLI.hpp"

Config::Config(int argc, char **argv)
{
    CLI::App app{"RayTraceDicom: Sub-second pencil beam dose calculation on GPU for adaptive proton therapy."};

    app.add_option<unsigned short>("--gpu_id",
                 gpu_id,
                 "ID of the GPU to use for simulation, starts from 0.");
                 //Only CUDA-capable GPUs are counted

    app.add_option("--ct_dir",
                 ct_dir,
                 "Patient CT directory. It must contain all the DICOM CT slices.")
    #ifndef WATER_CUBE_TEST
    ->required()
    #endif
    ;

    app.add_option("--rtplan",
                 rtplan,
                 "Path of the RTPLAN DICOM file to read.")
     #ifndef WATER_CUBE_TEST
    ->required()
    ->check(CLI::ExistingFile)
    #endif
    ;

    app.add_option("--output_directory",
                 output_directory,
                 "Directory where output will be stored.")
    ->required()
    ->check(CLI::ExistingDirectory)
    ;

    app.set_config("--config_file",
                 "",
                 "Specify a config file containing simulation parameters."
                 "The contents of the file are overriden by command line arguments.");

    try
    {
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError &e)
    {
        app.exit(e);
        exit = true;
    }

    std::cout << app.config_to_str(true, false) << std::endl;
}
