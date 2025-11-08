#pragma once

#include "tensor.hpp"

class TensorFactory
{
public:
    Tensor create(void *engine, backend::Tag backend_tag, int index);

};

extern TensorFactory tensorFactory;
