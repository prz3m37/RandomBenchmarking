#include "ConfigParser.h"

void ConfigParser::parseConfigFile()
{
    char split_char = '=';
    std::string line;
    std::ifstream myfile("configFile.txt");

    if (myfile.is_open())
    {
        int i = 0;
        while(std::getline(myfile, line))
        {
            std::stringstream   linestream(line);
            std::string         data;
            std::string         val1;
            int                 val2;
            std::getline(linestream, data, split_char);  // read up-to the first tab (discard tab).
            linestream >> val1;
            if (!val1.empty())
                *(this->params + i) = val1;
            else
                *(this->params + i) = "NO PARAM";
            i++;
        }
        setParams();
        myfile.close();
    }
    else 
        saveLog("[ERROR]: Unable to open configuration file");
};

void ConfigParser::setParams()
{
    this->gatesNumber = atoi(this->params[0].c_str());
    this->resultFilePath = this->params[1] + "_[" + utils.getCurrentTime() + "]_RB_RESULTS_FILE.txt";
    utils.saveLog("[INFO]: Configuration params loaded successfully");
}