use std::{
    cmp::{self, max, min, Ord},
    fmt::{self, Debug},
    ops::{self, Neg, Shl, Shr, Sub},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sign {
    PLUS,
    MINUS,
}

impl Sign {
    fn of<T>(v: T) -> Self
    where
        T: cmp::Ord + TryFrom<u8>,
        <T as TryFrom<u8>>::Error: fmt::Debug,
    {
        let zero: T = 0_u8.try_into().unwrap();
        if v >= zero {
            Sign::PLUS
        } else {
            Sign::MINUS
        }
    }
}

impl ops::Not for Sign {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            Sign::PLUS => Sign::MINUS,
            Sign::MINUS => Sign::PLUS,
        }
    }
}

impl ops::Mul for Sign {
    type Output = Self;
    
    fn mul(self, rhs: Self) -> Self::Output {
        if self == rhs {
            Sign::PLUS
        } else {
            Sign::MINUS
        }
    }
}

impl ops::MulAssign for Sign {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

use crate::digits::Digits;

#[derive(Debug, Clone, Eq)]
pub struct Integer {
    sign: Sign,
    value: Vec<u8>,
}

impl Integer {
    pub fn zero() -> Self {
        Integer { sign: Sign::PLUS, value: vec![] }
    }
    
    pub fn abs(self: &Self) -> Self {
        let Integer { value, .. } = &self;

        Integer {
            sign: Sign::PLUS,
            value: value.to_vec()
        }
    }

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
    
    fn cmp_abs(self: &Self, other: &Self) -> Option<cmp::Ordering> {
        use cmp::Ordering as Ord;

        let n1 = &self.value;
        let n2 = &other.value;

        let result = if n1.len() > n2.len() {
            Ord::Greater
        } else if n1.len() < n2.len() {
            Ord::Less
        } else {
            let it_1 = n1.iter().rev();
            let it_2 = n2.iter().rev();
            
            it_1.zip(it_2)
                .find_map(|(d1, d2)| {
                    if d1 > d2 {
                        Some(Ord::Greater)
                    } else if d1 < d2 {
                        Some(Ord::Less)
                    } else {
                        None
                    }
                })
                .map_or(Ord::Equal, |res| res)
        };

        Some(result)
    }

    fn add_abs(self: &Self, other: &Self) -> Self {
        let (v1, v2) = (&self.value, &other.value);

        let mut value: Vec<u8> = vec![];
        let mut carry: u8 = 0;

        for i in 0..max(v1.len(), v2.len()) {
            let n1 = v1.get(i).map_or(0, |&n| n);
            let n2 = v2.get(i).map_or(0, |&n| n);
            
            let (n, c) = match (n1, n2, carry) {
                (_, u8::MAX, 1) => (n1, 1),
                (u8::MAX, _, 1) => (n2, 1),
                _ => {
                    if n1 < u8::MAX - n2 - carry {
                        (n1 + n2 + carry, 0)
                    } else {
                        (n1 - (u8::MAX - n2 - carry) - 1, 1)
                    }
                }
            };

            value.push(n);
            carry = c;
        }

        if carry > 0 {
            value.push(carry);
        }

        Integer { sign: Sign::PLUS, value }
    }

    fn sub_abs(self: &Self, other: &Self) -> Self {
        let (sign, v1, v2) = if self.abs() > other.abs() {
            (Sign::PLUS, &self.value, &other.value)
        } else {
            (Sign::MINUS, &other.value, &self.value)
        };

        let mut value: Vec<u8> = vec![];
        let mut borrow: u8 = 0;    // 0 or 1

        for i in 0..max(v1.len(), v2.len()) {
            let n1 = v1.get(i).map_or(0, |&n| n);
            let n2 = v2.get(i).map_or(0, |&n| n);

            let [n1, n2] = [n1, n2].map(|n| n - min(n1, n2));

            let (n, b) = match (n1, n2, borrow) {
                (0, 0, 0) => (0, 0),
                (0, 0, 1) => (u8::MAX, 1),
                (0, _, 0) => (u8::MAX - n2 + 1, 1),
                (0, _, 1) => (u8::MAX - n2, 1),
                _ => (n1 - borrow, 0),
            };

            value.push(n);
            borrow = b;
        }

        while let Some(0) = value.last() {
            value.pop();
        }

        Integer { sign, value }
    }

    fn with_sign(self, sign: Sign) -> Self {
        Integer { sign, value: self.value }
    }
}

impl From<u8> for Integer {
    fn from(value: u8) -> Self {
        Integer {
            sign: Sign::PLUS,
            value: vec![value],
        }
    }
}

impl From<i8> for Integer {
    fn from(value: i8) -> Self {
        let sign = if value >= 0 {
            Sign::PLUS
        } else {
            Sign::MINUS
        };

        let v: u8 = value.abs().try_into().unwrap();

        Integer { sign, value: vec![v]}
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
        let abs: u16 = match v{
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

impl fmt::Display for Integer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Integer { sign, value } = self;

        let s = || -> String {
            let Digits(d) = value.into();
            dbg!(&d);
            d.into_iter().map(|n| n.to_string()).collect()
        };

        match (sign, value.len()) {
            (_, 0) => write!(f, "0"),
            (Sign::PLUS, _) => write!(f, "{}", s()),
            (Sign::MINUS, _) => write!(f, "-{}", s()),
        }
    }
}

impl cmp::PartialEq for Integer {
    fn eq(&self, other: &Self) -> bool {
        let (v1, v2) = (&self.value, &other.value);
        let (s1, s2) = (self.sign, other.sign);

        if v1.len() == 0 && v2.len() == 0 {
            true
        } else if s1 != s2 {
            false
        } else {
            v1 == v2
        }
    }
}

impl cmp::PartialOrd for Integer {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        use cmp::Ordering as Ord;
        
        match (self.sign, other.sign) {
            (Sign::PLUS, Sign::MINUS) => Some(Ord::Greater),
            (Sign::MINUS, Sign::PLUS) => Some(Ord::Less),
            (Sign::PLUS, Sign::PLUS) => self.cmp_abs(other),
            (Sign::MINUS, Sign::MINUS) => other.cmp_abs(self)
        }
    }
}

impl cmp::Ord for Integer {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.partial_cmp(other).expect(
            "partial_cmp for Integer must not return None."
        )
    }
}

impl ops::Neg for Integer {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let Integer { sign, value } = self;
        Integer { sign: !sign, value }
    }
}

impl ops::Add for Integer {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self.sign, rhs.sign) {
            (Sign::PLUS, Sign::PLUS) => Integer::add_abs(&self, &rhs),
            (Sign::MINUS, Sign::MINUS) => -Integer::add_abs(&self, &rhs),
            (Sign::PLUS, Sign::MINUS) => Integer::sub_abs(&self, &rhs),
            (Sign::MINUS, Sign::PLUS) => -Integer::sub_abs(&self, &rhs),
        }
    }
}

impl ops::Sub for Integer {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self.sign, rhs.sign) {
            (Sign::PLUS, Sign::MINUS) => Integer::add_abs(&self, &rhs),
            (Sign::MINUS, Sign::PLUS) => -Integer::add_abs(&self, &rhs),
            (Sign::PLUS, Sign::PLUS) => Integer::sub_abs(&self, &rhs),
            (Sign::MINUS, Sign::MINUS) => -Integer::sub_abs(&self, &rhs),
        }
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
        let res: Integer = 15_i8.into();
        let ans = Integer {
            sign: Sign::PLUS,
            value: vec![15]
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
mod test_integer_fmt {
    use super::*;

    #[test]
    fn zero() {
        assert_eq!("0", &format!("{}", Integer { sign: Sign::PLUS, value: vec![]}));
        assert_eq!("0", &format!("{}", Integer { sign: Sign::MINUS, value: vec![]}))
    }

    #[test]
    fn plus() {
        assert_eq!("123", &format!("{}", Integer { sign: Sign::PLUS, value: vec![123]}));
    }

    #[test]
    fn minus() {
        assert_eq!("-123", &format!("-{}", Integer { sign: Sign::PLUS, value: vec![123]}));
    }

    #[test]
    fn large() {
        assert_eq!("256", format!("{}", Integer { sign: Sign::PLUS, value: vec![0, 1] }))
    }
}

#[cfg(test)]
mod test_integer_cmp {
    use super::*;

    #[test]
    fn zero() {
        let v1 = Integer { sign: Sign::PLUS, value: vec![] };
        let v2 = Integer { sign: Sign::MINUS, value: vec![] };

        assert!(v1 == v2);
    }
    
    #[test]
    fn plus_minus() {
        let v1 = Integer { sign: Sign::PLUS, value: vec![123, 233] };
        let v2 = Integer { sign: Sign::MINUS, value: vec![123, 233] };

        assert!(v1 > v2);
    }

    #[test]
    fn minus_plus() {
        let v1 = Integer { sign: Sign::MINUS, value: vec![1, 12, 11] };
        let v2 = Integer { sign: Sign::PLUS, value: vec![1, 12, 11] };

        assert!(v1 < v2);
    }

    #[test]
    fn plus_greater() {
        let v1 = Integer { sign: Sign::PLUS, value: vec![123, 222] };
        let v2 = Integer { sign: Sign::PLUS, value: vec![123, 22] };

        assert!(v1 > v2);
    }

    #[test]
    fn plus_less() {
        let v1 = Integer { sign: Sign::PLUS, value: vec![123, 23] };
        let v2 = Integer { sign: Sign::PLUS, value: vec![123, 232] };

        assert!(v1 < v2);
    }

    #[test]
    fn minus_greater() {
        let v1 = Integer { sign: Sign::MINUS, value: vec![123, 123] };
        let v2 = Integer { sign: Sign::MINUS, value: vec![124, 123] };

        assert!(v1 > v2);
    }

    #[test]
    fn minus_less() {
        let v1 = Integer { sign: Sign::MINUS, value: vec![124, 123] };
        let v2 = Integer { sign: Sign::MINUS, value: vec![123, 123] };

        assert!(v1 < v2);
    }
}

#[cfg(test)]
mod test_integer_ops_utils {
    use super::Integer;

    #[test]
    fn add_abs() {
        let (v1, v2): (i32, i32) = (1000, -2000);
        let res = Integer::add_abs(&v1.into(), &v2.into());
        let ans = 3000.into();
        assert_eq!(res, ans);
    }
    
    #[test]
    fn sub_abs_res_pos() {
        let (v1, v2): (i32, i32) = (-50000, 10000);
        let res = Integer::sub_abs(&v1.into(), &v2.into());
        let ans = 40000.into();
        assert_eq!(res, ans);
    }

    #[test]
    fn sub_abs_res_neg() {
        let (v1, v2): (i32, i32) = (-10000, -50000);
        let res = Integer::sub_abs(&v1.into(), &v2.into());
        let ans: Integer = (-40000).into();
        assert_eq!(ans, res);
    }
}

#[cfg(test)]
mod test_integer_ops_add {
    use super::*;

    #[test]
    fn pos_pos() {
        let v1: Integer = 1000_i32.into();
        let v2: Integer = 10000_i32.into();
        let res = v1 + v2;
        let ans = 11000_i32.into();
        assert_eq!(res, ans);
    }
    
    #[test]
    fn neg_neg() {
        let v1: Integer = (-5555_i32).into();
        let v2: Integer = (-55_i32).into();
        let res = v1 + v2;
        let ans = (-5610_i32).into();
        assert_eq!(res, ans);
    }

    #[test]
    fn pos_neg_res_pos() {
        let v1: Integer = (5555_i32).into();
        let v2: Integer = (-55_i32).into();
        let res = v1 + v2;
        let ans = 5500_i32.into();
        assert_eq!(res, ans);
    }

    #[test]
    fn pos_neg_res_neg() {
        let v1: Integer = 1_i32.into();
        let v2: Integer = (-10000_i32).into();
        let res = v1 + v2;
        let ans = (-9999_i32).into();
        assert_eq!(res, ans);
    }

    #[test]
    fn neg_pos_res_pos() {
        let v1: Integer = (-55_i32).into();
        let v2: Integer = (5555_i32).into();
        let res = v1 + v2;
        let ans = 5500_i32.into();
        assert_eq!(res, ans);
    }

    #[test]
    fn neg_pos_res_neg() {
        let v1: Integer = (-5555_i32).into();
        let v2: Integer = (55_i32).into();
        let res = v1 + v2;
        let ans = (-5500_i32).into();
        assert_eq!(res, ans);
    }
}

#[cfg(test)]
mod test_integer_ops_sub {
    use super::*;

    #[test]
    fn pos_neg() {
        let v1: Integer = 500_000_000_i32.into();
        let v2: Integer = (-100_000_000_i32).into();
        let res = v1 - v2;
        let ans = 600_000_000_i32.into();
        assert_eq!(res, ans);
    }

    #[test]
    fn neg_pos() {
        let v1: Integer = (-100_000_000_i32).into();
        let v2: Integer = 500_000_000_i32.into();
        let res = v1 - v2;
        let ans = (-600_000_000_i32).into();
        assert_eq!(res, ans);
    }

    #[test]
    fn pos_pos_res_pos() {
        let v1: Integer = 100_000_000_i32.into();
        let v2: Integer = 1_i32.into();
        let res = v1 - v2;
        let ans = 99_999_999_i32.into();
        assert_eq!(res, ans);
    }

    #[test]
    fn pos_pos_res_neg() {
        let v1: Integer = 1_i32.into();
        let v2: Integer = 100_000_000_i32.into();
        let res = v1 - v2;
        let ans = (-99_999_999_i32).into();
        assert_eq!(res, ans);
    }

    #[test]
    fn neg_neg_res_pos() {
        let v1: Integer = (-1_i32).into();
        let v2: Integer = (-100_000_000_i32).into();
        let res = v1 - v2;
        let ans = 99_999_999_i32.into();
        assert_eq!(res, ans);
    }

    #[test]
    fn neg_neg_res_neg() {
        let v1: Integer = (-100_000_000_i32).into();
        let v2: Integer = (-1_i32).into();
        let res = v1 - v2;
        let ans = (-99_999_999_i32).into();
        assert_eq!(res, ans);
    }
}