#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include "Utils.h"

class ConfigParser{

    friend class Utils;

    protected:
        
        int gatesNumber;
        std::string resultFilePath;

    private:

        std::string params[2];
        void parseConfigFile();
        void setParams();
};

#endif /* CLIFFORD_GATE_H */