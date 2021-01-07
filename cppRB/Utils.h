#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include "ConfigParser.h"

class Utils{

    public: 
        void createLogFile();
        void closeLogFile();
    
    protected:
        saveLog(std::string);
        void saveResult(std::string);
        
    private:
        ConfigParser cfgParser;

        std::string logFilePath;
        std::ofstream logFile;
        std::ofstream resultsFile;

        void createResultFile();
        void closeResultFile();
        std::string getCurrentTime();
};

#endif /* UTILS_H */