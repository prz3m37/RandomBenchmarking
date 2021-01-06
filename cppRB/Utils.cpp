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
	this->logFile.open(this->logFilePath + "_" + 
    getCurrentTime() + "_RB_LOG_FILE.txt", std::ios_base::app);
    std::cout<<"[INFO]: Log file created \n"; 
}

void Utils::closeLogFile()
{
    std::cout<<"[INFO]: Log file closed \n"; 
    this->logFile.close();
}

void Utils::saveLog(std::string message)
{
    std::string currentTime = getCurrentTime();
	this->logFile << "[" + currentTime + "] " + message << "\n";
};

void Utils::parseConfigFile(std::string* params)
{
    char split_char = '=';
    std::string line;
    std::ifstream myfile("configFile.txt");
    int i = 0;

    if (myfile.is_open())
    {
        while(std::getline(myfile, line))
        {
            std::stringstream   linestream(line);
            std::string         data;
            std::string         val1;
            int                 val2;
            std::getline(linestream, data, split_char);  // read up-to the first tab (discard tab).
            linestream >> val1;
            if (!val1.empty())
                *(params + i) = val1;
            else
                *(params + i) = "NO PARAM";
            i++;
        }
        this->saveLog("[INFO]: Configuration params loaded successfully");
        myfile.close();
    }
    else 
        this->saveLog("[ERROR]: Unable to open configuration file");
};