#include "Utils.h"

std::string Utils::getCurrentTime()
{
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[20];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer,sizeof(buffer),"%d-%m-%Y %H:%M:%S",timeinfo);
    std::string time(buffer);
    return time;
};

void Utils::createLogFile()
{ 
    this->logFilePath = this->logFilePath + "_" + 
    getCurrentTime() + "_RB_LOG_FILE.txt";
	this->logFile.open(this->logFilePath, std::ios_base::app);
    std::cout<<"\n[INFO]: Random Benchmarking calculations started at: " + getCurrentTime() +  "\n\n"; 
    std::cout<<"[INFO]: Log file created \n"; 
};

void Utils::closeResultFile()
{ 
	this->resultsFile.close();
    saveLog("[INFO]: Results file closed");
};

void Utils::saveResult(std::string result)
{
    std::string currentTime = getCurrentTime();
	this->resultsFile << "[" + currentTime + "] " + result << "\n";
};

void Utils::createResultFile()
{ 
	this->resultsFile.open(cfgParser.resultFilePath, std::ios_base::app);
    saveLog("[INFO]: Results file created");
};

void Utils::closeLogFile()
{
    std::cout<<"[INFO]: Log file closed \n"; 
    std::cout<<"\n[INFO]: Random Benchmarking calculations ended at: " + getCurrentTime() +  "\n"; 
    this->logFile.close();
};

void Utils::saveLog(std::string message)
{
    std::string currentTime = getCurrentTime();
	this->logFile << "[" + currentTime + "] " + message << "\n";
};