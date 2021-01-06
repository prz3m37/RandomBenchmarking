#include "Utils.h"

int main()
{
    Utils utils;
    std::string params [2];
    utils.createLogFile();
    utils.parseConfigFile(params);
    utils.closeLogFile();
    return 1;
}