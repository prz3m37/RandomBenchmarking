#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>

class Utils{

    public: 
        void createLogFile();
        void closeLogFile();
        void parseConfigFile(std::string*);

    private:
        std::string logFilePath = "./";
        std::ofstream logFile;

        std::string getCurrentTime();
        void saveLog(std::string);
        void createResultFile();
};

#endif /* UTILS_H */