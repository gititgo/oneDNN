[![Build Status](https://travis-ci.org/herumi/xbyak_loongarch.png)](https://travis-ci.org/loongson/xbyak_loongarch)

# Xbyak_loongarch ; JIT assembler for loongArch CPUs by C++

## Abstract

Xbyak_loongarch is C++ header files which enables run-time assemble coding with the loongson instruction set of loongArch architecture.
Xbyak_loongarch is based on Xbyak developed for x86_64 CPUs by MITSUNARI Shigeo.

## Feature

* GNU assembler like syntax.
* Fully support LASX instructions.

### Requirement

Xbyak_loongarch uses no external library and it is written as standard C++ header files 
so that we believe Xbyak_loongarch works various environment with various compilers.

### News
Break backward compatibility:
- To link `libxbyak_loongarch.a` is always necessary.
- namespace `Xbyak` is renamed to `Xbyak_loongarch`.
- Some class are renamed (e.g. CodeGeneratorLoongArch -> CodeGenerator).
- L_loongarch() is renamed to L().
- use dd(uint32_t) instead of dw(uint32_t).

### Supported Compilers

Almost C++11 or later compilers for loongarch such as g++, clang++.

## Install

The command `make` builds `lib/libxbyak_loongarch.a`.

`make install` installs headers and a library into `/usr/local/` (default path).
Or add the location of the `xbyak_loongarch` directory to your compiler's include and link paths.

### Execution environment

You can execute programs using xbyak_loongarch on systems running on loongArch CPUs.

## How to make lib

```
make
```
makes `lib/libxbyak_loongarch.a`.

## How to use Xbyak_loongarch

Inherit `Xbyak::CodeGenerator` class and make the class method.
Make an instance of the class and get the function
pointer by calling `getCode()` and call it.
The following example 1) generates a JIT-ed function which simply adds two integer values passed as arguments and returns an integer value as a result,
and 2) calls the function. This example outputs "7" to STDOUT.

compile options:
- `-I <xbyak_loongarch dir>/xbyak_loongarch`.
- `-L <xbyak_loongarch dir>/lib -lxbyak_loongarch`.

```
#include "xbyak_loongarch.h"
using namespace Xbyak_loongarch;
class Generator : public CodeGenerator {
public:
  Generator() {
    Label L1, L2;
    L(L1);
    add(w0, w1, w0);
    cmp(w0, 13);
    b(EQ, L2);
    sub(w1, w1, 1);
    b(L1);
    L(L2);
    ret();
  }
};
int main() {
  Generator gen;
  gen.ready();
  auto f = gen.getCode<int (*)(int, int)>();
  std::cout << f(3, 4) << std::endl;
  return 0;
}
```

## Syntax
Synatx is similar to "AS" (GNU assembler).
Each LoongArch instruction is correspond to each function written in "xbyak_loongarch_mnemonic.h", we call it a **mnemonic function**.
Please refer files in sample/mnemonic_syntax directory for usage of mnemonic functions.
The below example shows correspondence between "AS" syntax and Xbyak_loongarch mnemonic functions.

```
"AS"                  Xbyak_loongarch
add w0, w0, w1  --> add(w0, w0, w1);
add x0, x0, x1  --> add(x0, x0, x1);
add x0, x0, 5   --> add(x0, x0, 5);
mov w0, 1       --> mov(w0, 1);
ret             --> ret();
```

### Mnemonic functions
Each **mnemonic function** corresponds to one LoongArch instruction.
Function name represents corresponding mnemonic of instruction.
Because **"and", "or", "not"** are reserved keywords C++ and **"."** can't be used in C++ function name,
the following special cases are exist.

|Mnemonic of instruction|Name of **mnemonic funciton**|
|----|----|
|and|and_|
|or|or_|
|not|not_|
|b.cond|b|

### Operand
This section explains operands, which are given to mnemonic functions as their arguments.

#### General purpose registers

As general purpose registers,
the following table shows example of mnemonic functions' arguments ("Instance name" column).

|Instance name|C++ class name|Remarks|
|----|----|----|
|w0, w1, ..., w30|WReg|32-bit general purpose registers|
|x0, x1, ..., x30|WReg|64-bit general purpose registers|
|wzr|WReg|32-bit zero register|
|xzr|XReg|64-bit zero register|
|wsp|WReg|32-bit stack pointer|
|sp|zXReg|64-bit stack pointer|

You can also use your original instance as mnemonic functions argumetns.
Please refer constructor of "C++ class name" in Xbyak_loongarch files.

```
WReg dstReg(0);
WReg srcReg0(1);
WReg srcReg1(2);

add(dstReg, srcReg0, srcReg1);  <--- (1)
add(w0, w1, w2);                <--- Output is same JIT code of (1)
```

##### SIMD/Floating point registers as scalar registers

As SIMD/Floating point registers with scalar use, 
the following table shows example of mnemonic functions' arguments ("Instance name" column).

|Instance name|C++ class name|Remarks|
|----|----|----|
|b0, b1, ..., b31|BReg|8-bit scalar registers|
|h0, h1, ..., h31|HReg|16-bit scalar registers|
|s0, s1, ..., s31|SReg|32-bit scalar registers|
|d0, d1, ..., d31|DReg|64-bit scalar registers|
|q0, q1, ..., q31|QReg|128-bit scalar registers|

You can also use your original instance as mnemonic functions argumetns.
Please refer constructor of "C++ class name" in Xbyak_loongarch files.

```
BReg dstReg(0);

mov(b0, v0.b[5]);       <--- (1)
mov(dstReg, v0.b[5]);   <--- Output is same JIT code of (1)

```

### Immediate values

You can use immediate values for arguments of mnemonic functions in the form that C++ syntax allows,
such as, "10", "-128", "0xFF", "1<<32", "3.5", etc.

Please care for range of values.
For example, "ADD (immediate)" instruction can receive signed 12-bit value
so that you have to ensure that the value passed to mnemonic function is inside the range.
Mnemonic functions of Xbyak_loongarch checks immediate values at runtime, 
and throws exception if it detects range over.

```
void genAddFunc() {
     int a = 1<<16;
     add(x0, x0, a);    <--- This mnemonic function throws exception at runtime.
     ret();
}
```

Some immediate values may not decided at coding time but runtime.
You should check the immediate values and handle them.

```
void gen_Summation_From_One_To_Parameter_Func(unsigned int N) {

    if(N < (1<<11)) {
        for(int i=0; i<N; i++) {
            add(x0, x0, N);
        }
        ret();
    } else {
        printf("Invalid parameter N=%d\n", N);
        printf("This function supports less than 2048.\n");
    }
}
```    

## Label

You can use "Label" to direct where branch instruction jump to.
The following example shows how to use "Label".

```
Label L1;           // (1), instance of Label class
 
mov(w4, w0); 
mov(w3, 0); 
mov(w0, 0); 
L(L1);              // (2), "L" function registers JIT code address of this position to label L1.
add(w0, w0, w4); 
add(w3, w3, 1); 
cmp(w2, w3); 
ccmp(w1, w3, 4, NE); 
bgt(L1);            // (3), set destination of branch instruction to the address stored in L1.
```

1. Prepare Label class instance.
1. Call the L function to register destination address to the instance.
1. Pass the instance to mnemonic functions correspond to branch instructions.

You can copy the address stored in "Label" instance by using `assignL` function.

```
Label L1,L2,L3; 
....
L(L1);               // JIT code address of this position is stored to L1.
....
L(L2);               // JIT code address of this position is stored to L1.
....
if (flag) { 
assignL(L3,L1);      // The address stored in L1 is copied to L3.
} else { 
assignL(L3,L2);      // The address stored in L1 is copied to L3.
} 
b(L3);               // If flag == true, branch destination is L1, otherwise L2.
```



## Code size
The default max size of JIT-ed code is 4096 bytes.
Specify the size in constructor of `CodeGenerator()` if necessary.

```
class Quantize : public Xbyak_loongarch::CodeGenerator {
public:
  Quantize()
    : CodeGenerator(8192)
  {
  }
  ...
};
```

## User allocated memory

You can make JIT-ed code on prepared memory.

Call `setProtectModeRE` yourself to change memory mode if using the prepared memory.

```
uint8_t alignas(4096) buf[8192]; // C++11 or later

struct Code : Xbyak_loongarch::CodeGenerator {
    Code() : Xbyak_loongarch::CodeGenerator(sizeof(buf), buf)
    {
        mov(rax, 123);
        ret();
    }
};

int main()
{
    Code c;
    c.setProtectModeRE(); // set memory to Read/Exec
    printf("%d\n", c.getCode<int(*)()>()());
}
```

**Note**: See [sample/test0.cpp](sample/test0.cpp).

### AutoGrow

If `AutoGrow` is specified in a constructor of `CodeGenerator`,
the memory region for JIT-ed code is automatically extended if needed.

Call `ready()` or `readyRE()` before calling `getCode()` to fix jump address.
```
struct Code : Xbyak_loongarch::CodeGenerator {
  Code()
    : Xbyak_loongarch::CodeGenerator(<default memory size>, Xbyak_loongarch::AutoGrow)
  {
     ...
  }
};
Code c;
// generate code for jit
c.ready(); // mode = Read/Write/Exec
```

**Note**:
* Don't use the address returned by `getCurr()` before calling `ready()` because it may be invalid address.

### Read/Exec mode
Xbyak_loongarch set Read/Write/Exec mode to memory to run JIT-ed code.
If you want to use Read/Exec mode for security, then specify `DontSetProtectRWE` for `CodeGenerator` and
call `setProtectModeRE()` after generating JIT-ed code.

```
struct Code : Xbyak_loongarch::CodeGenerator {
    Code()
        : Xbyak_loongarch::CodeGenerator(4096, Xbyak_loongarch::DontSetProtectRWE)
    {
        mov(eax, 123);
        ret();
    }
};

Code c;
c.setProtectModeRE();
```


Call `readyRE()` instead of `ready()` when using `AutoGrow` mode.
See [protect-re.cpp](sample/protect-re.cpp).

## How to pass arguments to JIT generated function
To be written...

## Macro
To be written...


## Sample
To be written...

* [add.cpp](sample/add.cpp) ; tiny sample
* [label.cpp](sample/label.cpp) ; label sample

## License

Copyright LOONGSON LIMITED 2019-2020

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Notice

* Loongson is a registered trademark of Loongson Limited (or its subsidiaries) in the US and/or elsewhere.
* Intel is a registered trademark of Intel Corporation (or its subsidiaries) in the US and/or elsewhere.



## Acknowledgement

We are grateful to MITSUNARI-san (Cybozu Labs, Inc.) for release Xbyak as an open source software and his advice for development of Xbyak_loongarch.

## History

|Date|Version|Remarks|
|----|----|----|
|December 9, 2019|0.9.0|First public release version.|


## Copyright

Copyright LOONGSON LIMITED 2019-2020
