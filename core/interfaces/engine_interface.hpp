#pragma once

#include <memory>

class IEngine {
public:
    virtual ~IEngine() = default;
    virtual void initialize(const std::string&) = 0; 
    virtual int infer(std::vector<float*> inputs, std::vector<float*> outputs) = 0;
    
};
