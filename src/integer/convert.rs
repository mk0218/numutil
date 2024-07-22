use std::{
    cmp::Ord,
    fmt::Debug,
    ops::{AddAssign, Neg, Shl, ShlAssign, Shr, Sub},
};

use super::{Integer, Sign};

impl Integer {
    fn from_uint<T>(v: T) -> Self
    where
        T: Copy + Ord + Sub + Debug,
        T: Shl<u8> + Shr<u8>,
        T: From<<T as Sub>::Output>,
        T: From<<T as Shr<u8>>::Output>,
        T: From<<T as Shl<u8>>::Output>,
        T: TryFrom<u8> + TryInto<u8>,
        <T as TryFrom<u8>>::Error: Debug,
        <T as TryInto<u8>>::Error: Debug,
    {
        if v < 0_u8.try_into().unwrap() {
            panic!("The argument for from_unit must be unsigned integer type.");
        }

        let mut v = v;
        let mut value: Vec<u8> = vec![];

        while v > 0_u8.try_into().unwrap() {
            let v_next: T = (v >> 8).try_into().unwrap();
            let v_curr: T = (v - (v_next << 8_u8).try_into().unwrap()).try_into().unwrap();
            value.push(v_curr.try_into().unwrap());
            v = v_next;
        }

        Integer { sign: Sign::PLUS, value }
    }

    fn eval_value<T>(&self) -> T
    where 
        T: AddAssign + ShlAssign,
        T: Into<Integer> + TryFrom<u8>,
        <T as TryFrom<u8>>::Error: Debug,
    {
        let mut sum: T = 0_u8.try_into().unwrap();

        for (i, &v) in self.value.iter().enumerate() {
            let mut v: T = v.try_into().unwrap();

            for _ in 0..i {
                v <<= 8_u8.try_into().unwrap();
            }

            sum += v;
        }

        sum
    }
    
    fn try_into_unsigned<T>(self, v_max: T) -> Result<T, TryIntoError>
    where
        T: AddAssign + ShlAssign,
        T: Into<Integer> + TryFrom<u8>,
        <T as TryFrom<u8>>::Error: Debug,
    {
        if self.sign == Sign::MINUS {
            Err(TryIntoError::NegIntoUnsigned)
        } else if self > v_max.into() {
            Err(TryIntoError::Overrflow)
        } else {
            Ok(self.eval_value())
        }
    }

    fn try_into_signed<T>(self, v_min: T, v_max: T) -> Result<T, TryIntoError>
    where
        T: Clone,
        T: AddAssign + ShlAssign + Neg<Output = T>,
        T: Into<Integer> + TryFrom<u8>,
        <T as TryFrom<u8>>::Error: Debug,
    {
        let i_min: Integer = v_min.clone().into();
        let i_max: Integer = v_max.clone().into();

        if self < i_min || self > i_max {
            Err(TryIntoError::Overrflow)
        } else if self == i_min {
            Ok(v_min)
        } else {
            let mut sum = self.eval_value::<T>();

            if self.sign == Sign::MINUS {
                sum = -sum;
            }

            Ok(sum)
        }
    }
}

impl From<u8> for Integer {
    fn from(v: u8) -> Self {
        Integer {
            sign: Sign::PLUS,
            value: vec![v],
        }
    }
}

impl From<i8> for Integer {
    fn from(v: i8) -> Self {
        let sign = Sign::of(v);
        let abs: u8 = match v {
            i8::MIN => 1 << 7,
            _ => v.abs().try_into().unwrap(),
        };

        Integer::from(abs).with_sign(sign)
    }
}

impl From<u16> for Integer {
    fn from(v: u16) -> Self {
        Integer::from_uint(v)
    }
}

impl From<i16> for Integer {
    fn from(v: i16) -> Self {
        let sign = Sign::of(v);
        let abs: u16 = match v {
            i16::MIN => 1 << 15,
            _ => v.abs().try_into().unwrap()
        };

        Integer::from_uint(abs).with_sign(sign)
    }
}

impl From<u32> for Integer {
    fn from(v: u32) -> Self {
        Integer::from_uint(v)
    }
}

impl From<i32> for Integer {
    fn from(v: i32) -> Self {
        let sign = Sign::of(v);
        let abs: u32 = match v {
            i32::MIN => 1 << 31,
            _ => v.abs().try_into().unwrap(),
        };

        Integer::from_uint(abs).with_sign(sign)
    }
}

impl From<u64> for Integer {
    fn from(v: u64) -> Self {
        Integer::from_uint(v)
    }
}

impl From<i64> for Integer {
    fn from(v: i64) -> Self {
        let sign = Sign::of(v);
        let abs: u64 = match v {
            i64::MIN => 1 << 63,
            _ => v.abs().try_into().unwrap(),
        };

        Integer::from_uint(abs).with_sign(sign)
    }
}

impl From<usize> for Integer {
    fn from(v: usize) -> Self {
        Integer::from_uint(v)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum TryIntoError {
    Overrflow,
    NegIntoUnsigned,
}

impl TryInto<u8> for Integer {
    type Error = TryIntoError;
    
    fn try_into(self) -> Result<u8, Self::Error> {
        self.try_into_unsigned(u8::MAX)
    }
}

impl TryInto<i8> for Integer {
    type Error = TryIntoError;
    
    fn try_into(self) -> Result<i8, Self::Error> {
        self.try_into_signed(i8::MIN, i8::MAX)
    }
}

impl TryInto<u16> for Integer {
    type Error = TryIntoError;

    fn try_into(self) -> Result<u16, Self::Error> {
        self.try_into_unsigned(u16::MAX)
    }
}

impl TryInto<i16> for Integer {
    type Error = TryIntoError;

    fn try_into(self) -> Result<i16, Self::Error> {
        self.try_into_signed(i16::MIN, i16::MAX)
    }
}

impl TryInto<u32> for Integer {
    type Error = TryIntoError;

    fn try_into(self) -> Result<u32, Self::Error> {
        self.try_into_unsigned(u32::MAX)
    }
}

impl TryInto<i32> for Integer {
    type Error = TryIntoError;

    fn try_into(self) -> Result<i32, Self::Error> {
        self.try_into_signed(i32::MIN, i32::MAX)
    }
}

impl TryInto<u64> for Integer {
    type Error = TryIntoError;

    fn try_into(self) -> Result<u64, Self::Error> {
        self.try_into_unsigned(u64::MAX)
    }
}

impl TryInto<i64> for Integer {
    type Error = TryIntoError;

    fn try_into(self) -> Result<i64, Self::Error> {
        self.try_into_signed(i64::MIN, i64::MAX)
    }
}

#[cfg(test)]
mod test_integer_from {
    use super::*;

    #[test]
    fn test_usize() {
        let res: Integer = 65535_usize.into();
        let ans = Integer {
            sign: Sign::PLUS,
            value: vec![255, 255]
        };
        assert_eq!(res, ans);
    }
    
    #[test]
    fn test_i8() {
        let res: Integer = i8::MIN.into();
        let ans = Integer {
            sign: Sign::MINUS,
            value: vec![128]
        };
        assert_eq!(res, ans);
    }

    #[test]
    fn test_u16() {
        let res: Integer = u16::MAX.into();
        let ans = Integer {
            sign: Sign::PLUS,
            value: vec![255, 255]
        };
        assert_eq!(ans, res);
    }

    #[test]
    fn test_i16() {
        let res: Integer = i16::MIN.into();
        let ans = Integer {
            sign: Sign::MINUS,
            value: vec![0, 128]
        };
        assert_eq!(ans, res);
    }

    #[test]
    fn test_u32() {
        let res: Integer = u32::MAX.into();
        let ans = Integer {
            sign: Sign::PLUS,
            value: vec![255, 255, 255, 255]
        };
        assert_eq!(ans, res);
    }

    #[test]
    fn test_i32() {
        let res: Integer = i32::MIN.into();
        let ans = Integer {
            sign: Sign::MINUS,
            value: vec![0, 0, 0, 128],
        };
        assert_eq!(ans, res);
    }
}

#[cfg(test)]
mod test_try_into {
    use super::*;

    #[test]
    fn test_u8_ok() {
        let v: Result<u8, _> = Integer {
            sign: Sign::PLUS,
            value: vec![255],
        }.try_into();

        assert_eq!(Ok(255), v)
    }

    #[test]
    fn test_u8_err_from_neg() {
        let v: Result<u8, _> = Integer {
            sign: Sign::MINUS,
            value: vec![255],
        }.try_into();

        assert_eq!(Err(TryIntoError::NegIntoUnsigned), v);
    }

    #[test]
    fn test_u8_err_overflow() {
        let v: Result<u8, _> = Integer {
            sign: Sign::PLUS,
            value: vec![0, 1],
        }.try_into();

        assert_eq!(Err(TryIntoError::Overrflow), v);
    }

    #[test]
    fn test_i8_ok() {
        let v: Result<i8, _> = Integer {
            sign: Sign::MINUS,
            value: vec![128],
        }.try_into();

        assert_eq!(Ok(-128), v);
    }

    #[test]
    fn test_i8_pos_err_overflow() {
        let v: Result<i8, _> = Integer {
            sign: Sign::PLUS,
            value: vec![128],
        }.try_into();

        assert_eq!(Err(TryIntoError::Overrflow), v);
    }

    #[test]
    fn test_i8_neg_err_overflow() {
        let v: Result<i8, _> = Integer {
            sign: Sign::MINUS,
            value: vec![129],
        }.try_into();

        assert_eq!(Err(TryIntoError::Overrflow), v);
    }

    #[test]
    fn test_u16_ok() {
        let v: Result<u16, _> = Integer {
            sign: Sign::PLUS,
            value: vec![255, 255],
        }.try_into();

        assert_eq!(Ok(u16::MAX), v)
    }

    #[test]
    fn test_u16_err_from_neg() {
        let v: Result<u16, _> = Integer {
            sign: Sign::MINUS,
            value: vec![255, 255],
        }.try_into();

        assert_eq!(Err(TryIntoError::NegIntoUnsigned), v);
    }

    #[test]
    fn test_u16_err_overflow() {
        let v: Result<u16, _> = Integer {
            sign: Sign::PLUS,
            value: vec![0, 0, 1],
        }.try_into();

        assert_eq!(Err(TryIntoError::Overrflow), v);
    }

    #[test]
    fn test_i16_ok() {
        let v: Result<i16, _> = Integer {
            sign: Sign::MINUS,
            value: vec![0, 128],
        }.try_into();

        assert_eq!(Ok(i16::MIN), v);
    }

    #[test]
    fn test_i16_pos_err_overflow() {
        let v: Result<i16, _> = Integer {
            sign: Sign::PLUS,
            value: vec![0, 128],
        }.try_into();

        assert_eq!(Err(TryIntoError::Overrflow), v);
    }

    #[test]
    fn test_i16_neg_err_overflow() {
        let v: Result<i16, _> = Integer {
            sign: Sign::MINUS,
            value: vec![0, 129],
        }.try_into();

        assert_eq!(Err(TryIntoError::Overrflow), v);
    }

    #[test]
    fn test_u32_ok() {
        let v: Result<u32, _> = Integer {
            sign: Sign::PLUS,
            value: vec![255, 255, 255, 255],
        }.try_into();

        assert_eq!(Ok(u32::MAX), v)
    }

    #[test]
    fn test_u32_err_from_neg() {
        let v: Result<u32, _> = Integer {
            sign: Sign::MINUS,
            value: vec![255, 255, 255, 255],
        }.try_into();

        assert_eq!(Err(TryIntoError::NegIntoUnsigned), v);
    }

    #[test]
    fn test_u32_err_overflow() {
        let v: Result<u32, _> = Integer {
            sign: Sign::PLUS,
            value: vec![0, 0, 0, 0, 1],
        }.try_into();

        assert_eq!(Err(TryIntoError::Overrflow), v);
    }

    #[test]
    fn test_i32_ok() {
        let v: Result<i32, _> = Integer {
            sign: Sign::MINUS,
            value: vec![0, 0, 0, 128],
        }.try_into();

        assert_eq!(Ok(i32::MIN), v);
    }

    #[test]
    fn test_i32_pos_err_overflow() {
        let v: Result<i32, _> = Integer {
            sign: Sign::PLUS,
            value: vec![0, 0, 0, 128],
        }.try_into();

        assert_eq!(Err(TryIntoError::Overrflow), v);
    }

    #[test]
    fn test_i32_neg_err_overflow() {
        let v: Result<i32, _> = Integer {
            sign: Sign::MINUS,
            value: vec![0, 0, 0, 129],
        }.try_into();

        assert_eq!(Err(TryIntoError::Overrflow), v);
    }

    #[test]
    fn test_u64_ok() {
        let v: Result<u64, _> = Integer {
            sign: Sign::PLUS,
            value: vec![255, 255, 255, 255, 255, 255, 255, 255],
        }.try_into();

        assert_eq!(Ok(u64::MAX), v)
    }

    #[test]
    fn test_u64_err_from_neg() {
        let v: Result<u64, _> = Integer {
            sign: Sign::MINUS,
            value: vec![255, 255, 255, 255, 255, 255, 255, 255],
        }.try_into();

        assert_eq!(Err(TryIntoError::NegIntoUnsigned), v);
    }

    #[test]
    fn test_u64_err_overflow() {
        let v: Result<u64, _> = Integer {
            sign: Sign::PLUS,
            value: vec![0, 0, 0, 0, 0, 0, 0, 0, 1],
        }.try_into();

        assert_eq!(Err(TryIntoError::Overrflow), v);
    }

    #[test]
    fn test_i64_ok() {
        let v: Result<i64, _> = Integer {
            sign: Sign::MINUS,
            value: vec![0, 0, 0, 0, 0, 0, 0, 128],
        }.try_into();

        assert_eq!(Ok(i64::MIN), v);
    }

    #[test]
    fn test_i64_pos_err_overflow() {
        let v: Result<i64, _> = Integer {
            sign: Sign::PLUS,
            value: vec![0, 0, 0, 0, 0, 0, 0, 128],
        }.try_into();

        assert_eq!(Err(TryIntoError::Overrflow), v);
    }

    #[test]
    fn test_i64_neg_err_overflow() {
        let v: Result<i64, _> = Integer {
            sign: Sign::MINUS,
            value: vec![0, 0, 0, 0, 0, 0, 0, 129],
        }.try_into();

        assert_eq!(Err(TryIntoError::Overrflow), v);
    }
}
