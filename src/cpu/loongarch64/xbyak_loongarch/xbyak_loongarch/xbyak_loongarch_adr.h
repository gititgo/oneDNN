#pragma once
/*******************************************************************************
 * Copyright 2019-2021 LOONGSON LIMITED
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "xbyak_loongarch_err.h"
#include "xbyak_loongarch_reg.h"

enum ShMod { LSL = 0, LSR = 1, ASR = 2, ROR = 3, MSL = 4, NONE = 5 };

enum ExtMod { UXTB = 0, UXTH = 1, UXTW = 2, UXTX = 3, SXTB = 4, SXTH = 5, SXTW = 6, SXTX = 7, UXT = 8, SXT = 9, MUL = 10, MUL_VL = 11, EXT_LSL = 12 };

enum AdrKind {
  // for v8
  BASE_ONLY = 1,     // base register only
  BASE_IMM = 1 << 1, // base plus offset (immediate)
  BASE_REG = 1 << 2, // base plus offset (register)
  BASE_EXT = 1 << 3, // base plus offset (extend register)
  PRE = 1 << 4,      // pre-indexed
  POST_IMM = 1 << 5, // post-indexed (immediate)
  POST_REG = 1 << 6, // post-indexed (register)

  // for SVE
  SC_SC = 1 << 7,        // scalar base plus scalar index
  SC_IMM = 1 << 8,       // scalar base plus immediate
  SC_64VEC = 1 << 9,     // scalar base plus 64-bit vector index
  SC_32VEC64E = 1 << 10, // scalar base plus 32-bit vecotr index (64-bit element)
  SC_32VEC32E = 1 << 11, // scalar base plus 32-bit vecotr index (32-bit element)
  VEC_IMM64E = 1 << 12,  // vector base plus immediate offset (64-bit element)
  VEC_IMM32E = 1 << 13,  // vector base plus immediate offset (32-bit element)

  // for SVE address generator
  VEC_PACK = 1 << 14,   // vector (packed)
  VEC_UNPACK = 1 << 15, // vector (unpacked)
};

class Adr {
  AdrKind kind_;

protected:
  ExtMod trans(ExtMod org) { return ((org == UXT) ? UXTW : (org == SXT) ? SXTW : org); }

  ExtMod trans(const RReg &rm, ExtMod org) {
    if (org == SXT) {
      return (rm.getBit() == 64) ? SXTX : SXTW;
    } else if (org == UXT) {
      return (rm.getBit() == 64) ? UXTX : UXTW;
    }
    return org;
  }

public:
  explicit Adr(AdrKind kind) : kind_(kind) {}
  AdrKind getKind() { return kind_; }
};

// Pre-indexed
class AdrPreImm : public Adr {
  XReg xn_;
  int32_t imm_;

public:
  explicit AdrPreImm(const XReg &xn, int32_t imm) : Adr(PRE), xn_(xn), imm_(imm) {}
  const XReg &getXn() const { return xn_; }
  int32_t getImm() const { return imm_; }
};

// Pos-indexed (immediate offset)
class AdrPostImm : public Adr {
  XReg xn_;
  int32_t imm_;

public:
  explicit AdrPostImm(const XReg &xn, int32_t imm) : Adr(POST_IMM), xn_(xn), imm_(imm) {}
  const XReg &getXn() const { return xn_; }
  int32_t getImm() const { return imm_; }
};

// Pos-indexed (register offset)
class AdrPostReg : public Adr {
  XReg xn_;
  XReg xm_;

public:
  explicit AdrPostReg(const XReg &xn, const XReg &xm) : Adr(POST_REG), xn_(xn), xm_(xm) {}
  const XReg &getXn() const { return xn_; }
  const XReg &getXm() const { return xm_; }
};

// base only
class AdrNoOfs : public Adr {
  XReg xn_;

public:
  explicit AdrNoOfs(const XReg &xn) : Adr(BASE_ONLY), xn_(xn) {}
  const XReg &getXn() const { return xn_; }
};

// base plus offset (signed immediate)
class AdrImm : public Adr {
  XReg xn_;
  int32_t imm_;

public:
  explicit AdrImm(const XReg &xn, int32_t imm) : Adr(BASE_IMM), xn_(xn), imm_(imm) {}
  AdrImm(const AdrNoOfs &a) : Adr(BASE_IMM), xn_(a.getXn()), imm_(0) {}
  const XReg &getXn() const { return xn_; }
  int32_t getImm() const { return imm_; }
};

// base plus offset (unsigned immediate)
class AdrUimm : public Adr {
  XReg xn_;
  uint32_t uimm_;

public:
  explicit AdrUimm(const XReg &xn, uint32_t uimm) : Adr(BASE_IMM), xn_(xn), uimm_(uimm) {}
  AdrUimm(const AdrNoOfs &a) : Adr(BASE_IMM), xn_(a.getXn()), uimm_(0) {}
  AdrUimm(const AdrImm &a) : Adr(BASE_IMM), xn_(a.getXn()), uimm_(a.getImm()) {}
  const XReg &getXn() const { return xn_; }
  uint32_t getImm() const { return uimm_; }
};

// base size_t offset (unsigned immediate) for binary_injector
class Address : public Adr {
  XReg xn_;
  size_t offset_;
  bool broadcast_;

public:
  explicit Address(const XReg &xn, size_t offset, bool broadcast = false) : Adr(BASE_IMM), xn_(xn), offset_(offset), broadcast_(broadcast) {}
  const XReg &getXReg() const { return xn_; }
  uint32_t getOffset() const { return offset_; }
  uint32_t getIdx() const { return xn_.getIdx(); }
  bool operator==(const Address& rhs) const
  {
    return getIdx() == rhs.getIdx() && getOffset() == rhs.getOffset() && getBroadcast() == rhs.getBroadcast();
  }
  bool operator!=(const Address& rhs) const { return !operator==(rhs); }
  bool getBroadcast() const { return broadcast_; }
};

// base plus offset (register)
class AdrReg : public Adr {
  XReg xn_;
  XReg xm_;
  ShMod mod_;
  uint32_t sh_;
  bool init_sh_;

public:
  explicit AdrReg(const XReg &xn, const XReg &xm, ShMod mod, uint32_t sh) : Adr(BASE_REG), xn_(xn), xm_(xm), mod_(mod), sh_(sh), init_sh_(true) {}
  explicit AdrReg(const XReg &xn, const XReg &xm, ShMod mod = LSL) : Adr(BASE_REG), xn_(xn), xm_(xm), mod_(mod), sh_(0), init_sh_(false) {}
  const XReg &getXn() const { return xn_; }
  const XReg &getXm() const { return xm_; }
  ShMod getMod() const { return mod_; }
  uint32_t getSh() const { return sh_; }
  bool getInitSh() const { return init_sh_; }
};

// base plus offset (extended register)
class AdrExt : public Adr {
  XReg xn_;
  RReg rm_;
  ExtMod mod_;
  uint32_t sh_;
  bool init_sh_;

public:
  explicit AdrExt(const XReg &xn, const RReg &rm, ExtMod mod, uint32_t sh) : Adr(BASE_EXT), xn_(xn), rm_(rm), mod_(trans(rm, mod)), sh_(sh), init_sh_(true) {}
  explicit AdrExt(const XReg &xn, const RReg &rm, ExtMod mod) : Adr(BASE_EXT), xn_(xn), rm_(rm), mod_(trans(rm, mod)), sh_(0), init_sh_(false) {}
  const XReg &getXn() const { return xn_; }
  const RReg &getRm() const { return rm_; }
  ExtMod getMod() const { return mod_; }
  uint32_t getSh() const { return sh_; }
  bool getInitSh() const { return init_sh_; }
};

///////////////////// for SVE //////////////////////////////////////////

// Scalar plus scalar
class AdrScSc : public Adr {
  XReg xn_;
  XReg xm_;
  ShMod mod_;
  uint32_t sh_;
  bool init_mod_;

public:
  explicit AdrScSc(const XReg &xn, const XReg &xm, ShMod mod, uint32_t sh) : Adr(SC_SC), xn_(xn), xm_(xm), mod_(mod), sh_(sh), init_mod_(true) {}
  explicit AdrScSc(const XReg &xn, const XReg &xm) : Adr(SC_SC), xn_(xn), xm_(xm), mod_(LSL), sh_(0), init_mod_(false) {}
  // AdrScSc(const AdrNoOfs &a) :Adr(SC_SC), xn_(a.getXn()), xm_(XReg(31)),
  // mod_(LSL), sh_(0) {}
  AdrScSc(const AdrReg &a) : Adr(SC_SC), xn_(a.getXn()), xm_(a.getXm()), mod_(a.getMod()), sh_(a.getSh()) {}
  const XReg &getXn() const { return xn_; }
  const XReg &getXm() const { return xm_; }
  uint32_t getSh() const { return sh_; }
  ShMod getMod() const { return mod_; }
  bool getInitMod() const { return init_mod_; }
};

// Scalar plus immediate
class AdrScImm : public Adr {
  XReg xn_;
  int32_t simm_;
  ExtMod mod_;

public:
  explicit AdrScImm(const XReg &xn, int32_t simm = 0, ExtMod mod = MUL_VL) : Adr(SC_IMM), xn_(xn), simm_(simm), mod_(trans(mod)) {}
  // AdrScImm(const AdrNoOfs &a) :Adr(SC_IMM), xn_(a.getXn()), simm_(0),
  // mod_(MUL_VL) {}
  AdrScImm(const AdrImm &a) : Adr(SC_IMM), xn_(a.getXn()), simm_(a.getImm()), mod_(MUL_VL) {}
  const XReg &getXn() const { return xn_; }
  int32_t getSimm() const { return simm_; }
  ExtMod getMod() const { return mod_; }
};

AdrNoOfs ptr(const XReg &xn);
AdrImm ptr(const XReg &xn, int32_t imm);
AdrUimm ptr(const XReg &xn, uint32_t uimm);
AdrReg ptr(const XReg &xn, const XReg &xm);
AdrReg ptr(const XReg &xn, const XReg &xm, ShMod mod, uint32_t sh);
AdrReg ptr(const XReg &xn, const XReg &xm, ShMod mod);
AdrExt ptr(const XReg &xn, const RReg &rm, ExtMod mod, uint32_t sh);
AdrExt ptr(const XReg &xn, const RReg &rm, ExtMod mod);
AdrPreImm pre_ptr(const XReg &xn, int32_t imm);
AdrPostImm post_ptr(const XReg &xn, int32_t imm);
AdrPostReg post_ptr(const XReg &xn, XReg xm);
AdrScImm ptr(const XReg &xn, int32_t simm, ExtMod mod);

inline AdrNoOfs ptr(const XReg &xn) { return AdrNoOfs(xn); }

inline AdrImm ptr(const XReg &xn, int32_t imm) { return AdrImm(xn, imm); }

inline AdrUimm ptr(const XReg &xn, uint32_t uimm) { return AdrUimm(xn, uimm); }

inline Address ptr_a(const XReg &xn, size_t offset) { return Address(xn, offset); }

inline Address ptr_b(const XReg &xn, size_t offset) { return Address(xn, offset, true); }

inline AdrReg ptr(const XReg &xn, const XReg &xm) { return AdrReg(xn, xm); }

inline AdrReg ptr(const XReg &xn, const XReg &xm, ShMod mod, uint32_t sh) { return AdrReg(xn, xm, mod, sh); }

inline AdrReg ptr(const XReg &xn, const XReg &xm, ShMod mod) { return AdrReg(xn, xm, mod); }

inline AdrExt ptr(const XReg &xn, const RReg &rm, ExtMod mod, uint32_t sh) { return AdrExt(xn, rm, mod, sh); }

inline AdrExt ptr(const XReg &xn, const RReg &rm, ExtMod mod) { return AdrExt(xn, rm, mod); }

inline AdrPreImm pre_ptr(const XReg &xn, int32_t imm) { return AdrPreImm(xn, imm); }

inline AdrPostImm post_ptr(const XReg &xn, int32_t imm) { return AdrPostImm(xn, imm); }

inline AdrPostReg post_ptr(const XReg &xn, XReg xm) { return AdrPostReg(xn, xm); }

inline AdrScImm ptr(const XReg &xn, int32_t simm, ExtMod mod) { return AdrScImm(xn, simm, mod); }

