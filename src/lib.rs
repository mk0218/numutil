use std::{
    cmp::{self, max, min},
    fmt,
    ops
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sign {
    PLUS,
    MINUS,
}

impl Sign {
    fn of(v: i32) -> Self {
        if v >= 0 {
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

use digits::Digits;

#[derive(Debug, Eq)]
struct Integer {
    sign: Sign,
    value: Vec<u32>,
}

/// The maximum u32 value among power of 10s.\
/// u32::MAX is 4,294,967,295, so it the max value is 1,000,000,000.
const MAX_POW_10_VALUE: u32 = 1_000_000_000;
/// Exponent for MAX_POW_10_VALUE.
const MAX_POW_10_EXP: u32 = 9;

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

        let mut value: Vec<u32> = vec![];
        let mut carry: u32 = 0;

        for i in 0..max(v1.len(), v2.len()) {
            let n1 = v1.get(i).map_or(0, |&n| n);
            let n2 = v2.get(i).map_or(0, |&n| n);
            
            let (n, c) = match (n1, n2, carry) {
                (_, u32::MAX, 1) => (n1, 1),
                (u32::MAX, _, 1) => (n2, 1),
                _ => {
                    if n1 < u32::MAX - n2 - carry {
                        (n1 + n2 + carry, 0)
                    } else {
                        (n1 - (u32::MAX - n2 - carry) - 1, 1)
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

        let mut value: Vec<u32> = vec![];
        let mut borrow: u32 = 0;    // 0 or 1

        for i in 0..max(v1.len(), v2.len()) {
            let n1 = v1.get(i).map_or(0, |&n| n);
            let n2 = v2.get(i).map_or(0, |&n| n);

            let [n1, n2] = [n1, n2].map(|n| n - min(n1, n2));

            let (n, b) = match (n1, n2, borrow) {
                (0, 0, 0) => (0, 0),
                (0, 0, 1) => (u32::MAX, 1),
                (0, _, 0) => (u32::MAX - n2 + 1, 1),
                (0, _, 1) => (u32::MAX - n2, 1),
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

impl From<i32> for Integer {
    fn from(v: i32) -> Self {
        match v {
            0 => Integer::zero(),
            i32::MIN => Integer {
                sign: Sign::MINUS,
                value: vec![1 << 31],
            },
            _ => Integer {
                sign: Sign::of(v),
                value: vec![v.abs().try_into().unwrap()],
            }
        }
    }
}

impl From<u32> for Integer {
    fn from(v: u32) -> Self {
        match v {
            0 => Integer::zero(),
            _ => Integer {
                sign: Sign::PLUS,
                value: vec![v],
            }
            
        }
    }
}

impl From<&str> for Integer {
    fn from(s: &str) -> Self {
        // TODO: Support for various number formats
        // TODO: Validate string.
        let (sign, s) = match s.chars().nth(0) {
            Some('-') => (Sign::MINUS, &s[1..]),
            Some('+') => (Sign::PLUS, &s[1..]),
            _ => (Sign::PLUS, s),
        };

        let chunk_size: usize = MAX_POW_10_EXP.try_into().expect(
            "16-bit architecture is not supported."
        );

        let mut chunks: Vec<&str> = vec![];

        let mut i: usize = 0;
        let mut j: usize = s.len() % chunk_size;

        while j <= s.len() {
            chunks.push(&s[i..j]);
            (i, j) = (j, j + chunk_size);
        }

        let from_chunk = |s: &str, i: usize| -> Integer {
            let uint = s
                .parse::<u32>()
                .expect("str validation not yet implemented");
            
            (uint * MAX_POW_10_VALUE.pow(i.try_into().unwrap())).into()
        };

        chunks
            .into_iter()
            .rev()
            .enumerate()
            .fold(Integer::zero(), |acc, (i, s)| {
                acc + from_chunk(s, i)
            })
            .with_sign(sign)
    }
}

impl fmt::Display for Integer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Integer { sign, value } = self;

        let s = || -> String {
            let mut d: Digits = Digits(vec![]);

            for (i, &v) in value.iter().enumerate() {
                let d1 = Digits::from_pow_2(i * 32);
                let d2: Digits = v.into();
                let d3 = d1 * d2;
                d += d3;
            }

            d.0.into_iter().rev().map(|n| n.to_string()).collect()
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
mod test_integer_from_i32 {
    use super::*;

    #[test]
    fn zero() {
        let ans = Integer::zero();
        assert_eq!(ans, 0_i32.into());
    }

    #[test]
    fn pos() {
        let ans = Integer { sign: Sign::PLUS, value: vec![1212] };
        assert_eq!(ans, 1212.into());
    }

    #[test]
    fn neg() {
        let ans = Integer { sign: Sign::MINUS, value: vec![1212] };
        assert_eq!(ans, (-1212).into());
    }

    #[test]
    fn max() {
        let abs: u32 = i32::MAX.try_into().unwrap();
        assert_eq!(abs, 2_147_483_647);

        let ans = Integer {
            sign: Sign::PLUS,
            value: vec![abs]
        };

        assert_eq!(ans, i32::MAX.into());
    }

    #[test]
    fn min() {
        let abs = 1 << 31;
        assert_eq!(abs, 2_147_483_648);

        let ans = Integer {
            sign: Sign::MINUS,
            value: vec![abs],
        };

        assert_eq!(ans, i32::MIN.into());
    }
}

#[cfg(test)]
mod test_integer_from_str {
    use super::*;    

    #[test]
    fn zero() {
        let ans = Integer::zero();
        assert_eq!(ans, "0".into());
    }

    #[test]
    fn zero_minus() {
        let ans = Integer::zero();
        assert_eq!(ans, "-0".into());
    }

    #[test]
    fn pos_abs_small_unsigned() {
        let ans: Integer = 100.into();
        assert_eq!(ans, "100".into());
    }

    #[test]
    fn pos_abs_small_signed() {
        let ans: Integer = 100.into();
        assert_eq!(ans, "+100".into());
    }

    #[test]
    fn neg_abs_small() {
        let ans: Integer = (-100).into();
        assert_eq!(ans, "-100".into());
    }

    #[test]
    fn pos_abs_large_unsigned() {
        let s = "4294967296";
        let ans = Integer {
            sign: Sign::PLUS,
            value: vec![4294967295, 1],
        };
        assert_eq!(ans, s.into());
    }

    #[test]
    fn pos_abs_large_signed() {
        let s = "+4294967296";
        let ans = Integer {
            sign: Sign::PLUS,
            value: vec![1, 1],
        };
        assert_eq!(ans, s.into());
    }

    #[test]
    fn neg_abs_large() {
        let s = "-4294967297";
        let ans = Integer {
            sign: Sign::PLUS,
            value: vec![2, 1],
        };
        assert_eq!(ans, s.into());
    }
}

#[cfg(test)]
mod test_integer_ops {
    use super::*;

    #[test]
    fn abs_pos() {
        let v: Integer = 1234.into();
        let ans: Integer = 1234.into();
        assert_eq!(ans, v.abs());
    }

    #[test]
    fn abs_neg() {
        let v: Integer = (-1234).into();
        let ans: Integer = 1234.into();
        assert_eq!(ans, v.abs());
    }

    #[test]
    fn add_abs() {
        let v1: Integer = (-9899).into();
        let v2: Integer = 119.into();
        let ans: Integer = 10_018.into();
        assert_eq!(ans, Integer::add_abs(&v1, &v2));
    }

    #[test]
    fn add_abs_large() {
        let v1: Integer = i32::MIN.into();
        let v2: Integer = i32::MIN.into();

        let ans: Vec<u32> = vec![0, 1];
        let Integer { value: res, .. } = Integer::add_abs(&v1, &v2);

        assert_eq!(ans, res);
    }

    #[test]
    fn sub_abs_pos() {
        let v1: Integer = (-9000).into();
        let v2: Integer = 900.into();
        let ans: Integer = 8100.into();
        assert_eq!(ans, Integer::sub_abs(&v1, &v2));
    }

    #[test]
    fn sub_abs_neg() {
        let v1: Integer = (-22).into();
        let v2: Integer = 2231.into();
        let ans: Integer = (-2209).into();
        assert_eq!(ans, Integer::sub_abs(&v1, &v2));
    }

    #[test]
    fn sub_abs_pos_abs_large() {
        let v1: Integer = Integer { sign: Sign::MINUS, value: vec![1, 1] };
        let v2: Integer = Integer { sign: Sign::MINUS, value: vec![0, 1, 1] };
        let ans = Integer { sign: Sign::MINUS, value: vec![u32::MAX, u32::MAX] };
        
        assert_eq!(format!("{}", ans), "-18446744073709551615");
        assert_eq!(ans, Integer::sub_abs(&v1, &v2));
    }
    
    #[test]
    fn add_small_pos_pos() {
        let v1: Integer = 510.into();
        let v2: Integer = 10621.into();
        let ans: Integer = 11131.into();

        assert_eq!(ans, v1 + v2);
    }

    #[test]
    fn add_small_neg_neg() {
        let v1: Integer = (-55_555).into();
        let v2: Integer = (-555_555).into();
        let ans: Integer = (-611_110).into();

        assert_eq!(ans, v1 + v2);
    }

    #[test]
    fn add_small_pos_neg_pos() {
        let v1: Integer = 10000.into();
        let v2: Integer = (-100).into();
        let ans: Integer = 9900.into();

        assert_eq!(ans, v1 + v2);
    }

    #[test]
    fn add_small_pos_neg_neg() {
        let v1: Integer = 1001.into();
        let v2: Integer = (-10000).into();
        let ans: Integer = (-8999).into();

        assert_eq!(ans, v1 + v2);
    }

    #[test]
    fn add_small_neg_pos_pos() {
        let v1: Integer = (-100).into();
        let v2: Integer = 10000.into();
        let ans: Integer = 9900.into();

        assert_eq!(ans, v1 + v2);
    }

    #[test]
    fn add_small_neg_pos_neg() {
        let v1: Integer = (-10000).into();
        let v2: Integer = 1001.into();
        let ans: Integer = (-8999).into();

        assert_eq!(ans, v1 + v2);
    }

    #[test]
    fn add_large_pos_pos_fmt() {
        let value = 1_u32 << 31;

        assert_eq!(value, 2_147_483_648);

        let v1: Integer = value.into();
        let v2: Integer = value.into();
        let ans = "4294967296";

        assert_eq!(ans, &format!("{}", v1 + v2));
    }

    #[test]
    fn add_large_neg_neg_fmt() {
        let value = i32::MIN;

        let v1: Integer = value.into();
        let v2: Integer = value.into();
        let ans = "-4294967296";

        assert_eq!(ans, &format!("{}", v1 + v2));
    }

    #[test]
    fn add_large_pos_neg_fmt() {
        todo!();
    }

    #[test]
    fn add_large_neg_pos_fmt() {
        todo!();
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
        assert_eq!("-12312", &format!("-{}", Integer { sign: Sign::PLUS, value: vec![12312]}));
    }

    #[test]
    fn large() {
        assert_eq!("4294967296", format!("{}", Integer { sign: Sign::PLUS, value: vec![0, 1] }))
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
    fn eq_plus() {
        let v1 = Integer { sign: Sign::PLUS, value: vec![123123, 2323] };
        let v2 = Integer { sign: Sign::PLUS, value: vec![123123, 2323] };

        assert!(v1 == v2);
    }
    
    #[test]
    fn eq_minus() {
        let v1 = Integer { sign: Sign::MINUS, value: vec![123123, 2323] };
        let v2 = Integer { sign: Sign::MINUS, value: vec![123123, 2323] };

        assert!(v1 == v2);
    }
    
    #[test]
    fn plus_minus() {
        let v1 = Integer { sign: Sign::PLUS, value: vec![123123, 2323] };
        let v2 = Integer { sign: Sign::MINUS, value: vec![1, 231, 12112] };

        assert!(v1 > v2);
    }

    #[test]
    fn minus_plus() {
        let v1 = Integer { sign: Sign::MINUS, value: vec![1, 12, 999] };
        let v2 = Integer { sign: Sign::PLUS, value: vec![121, 242, 2313] };

        assert!(v1 < v2);
    }

    #[test]
    fn greater_plus() {
        let v1 = Integer { sign: Sign::PLUS, value: vec![123123, 23231] };
        let v2 = Integer { sign: Sign::PLUS, value: vec![123123, 2323] };

        assert!(v1 > v2);
    }

    #[test]
    fn less_plus() {
        let v1 = Integer { sign: Sign::PLUS, value: vec![123123, 23231] };
        let v2 = Integer { sign: Sign::PLUS, value: vec![123123, 232312] };

        assert!(v1 < v2);
    }

    #[test]
    fn greater_minus() {
        let v1 = Integer { sign: Sign::MINUS, value: vec![123123, 23231] };
        let v2 = Integer { sign: Sign::MINUS, value: vec![123123, 232312] };

        assert!(v1 > v2);
    }

    #[test]
    fn less_minus() {
        let v1 = Integer { sign: Sign::MINUS, value: vec![123123, 23231] };
        let v2 = Integer { sign: Sign::MINUS, value: vec![123123, 2323] };

        assert!(v1 < v2);
    }
}

mod digits {
    use std::cmp::max;
    use std::ops;

    #[derive(PartialEq, Debug)]
    pub struct Digits(pub Vec<u8>);

    fn add(vec_1: &Vec<u8>, vec_2: &Vec<u8>) -> Vec<u8> {
        let mut result: Vec<u8> = vec![];
        let mut carry = 0;
    
        let maxlen = max(vec_1.len(), vec_2.len());
    
        for i in 0..maxlen {
            let v1 = vec_1.get(i).map_or(0_u8, |&v| v);
            let v2 = vec_2.get(i).map_or(0_u8, |&v| v);
            result.push(v1 + v2 + carry);
            carry = result[i] / 10;
            result[i] %= 10;
        }
    
        while carry > 0 {
            result.push(carry % 10);
            carry /= 10;
        }
    
        result  
    }

    fn prod(values: &Vec<u8>, scalar: u8) -> Vec<u8> {
        let mut result: Vec<u8> = vec![];
        let mut carry = 0;

        for (i, v) in values.iter().enumerate() {
            result.push(v * scalar + carry);
            carry = result[i] / 10;
            result[i] %= 10;
        }

        while carry > 0 {
            result.push(carry % 10);
            carry /= 10;
        }

        result
    }

    impl Digits {
        pub fn from_pow_2(p: usize) -> Digits {
            let mut d1 = Digits(vec![1]);
            let mut p = p;

            while p >= 32 {
                d1 = d1 * Digits(vec![6, 9, 2, 7, 6, 9, 4, 9, 2, 4]);
                p -= 32;
            }

            let p: u32 = p.try_into().unwrap();
            let d2: Digits = 10_u32.pow(p).into();

            d1 * d2
        }
    }
    
    impl ops::Add for Digits {
        type Output = Digits;

        fn add(self, rhs: Self) -> Self::Output {
            Digits(add(&self.0, &rhs.0))
        }
    }

    impl ops::AddAssign for Digits {
        fn add_assign(&mut self, rhs: Self) {
            self.0 = add(&self.0, &rhs.0);
        }
    }

    impl ops::Mul for Digits {
        type Output = Digits;

        fn mul(self, rhs: Self) -> Self::Output {
            let values = rhs.0.iter().map(|&s| prod(&self.0, s));
            let mut result: Vec<u8> = vec![];

            for (mut i, mut v) in values.enumerate() {
                while i > 0 {
                    v.insert(0, 0);
                    i -= 1;
                }

                result = add(&result, &v);
            }

            Digits(result)
        }
    }

    impl From<u32> for Digits {
        fn from(value: u32) -> Self {
            let mut v = value;
            let mut d: Vec<u8> = vec![];

            while v > 0 {
                d.push((v % 10).try_into().unwrap());
                v /= 10;
            }

            Digits(d)
        }
    }
    
#[cfg(test)]
    mod test_digits {
        use super::*;

        #[test]
        fn test_util_prod() {
            let input: Vec<u8> = vec![9, 9, 9, 9, 9, 9, 9, 9];
            let ans: Vec<u8> = vec![1, 9, 9, 9, 9, 9, 9, 9, 8];
            assert_eq!(ans, prod(&input, 9))
        }

        #[test]
        fn test_util_add() {
            let input_1: Vec<u8> = vec![5, 5, 5, 5];
            let input_2: Vec<u8> = vec![6, 4, 5];
            // 5555 + 546 = 6101
            let ans: Vec<u8> = vec![1, 0, 1, 6];
            assert_eq!(ans, add(&input_1, &input_2));
        }

        #[test]
        fn test_digits_add() {
            let d1 = Digits(vec![0, 0, 1]);
            let d2 = Digits(vec![2, 2, 2, 2]);
            let ans = Digits(vec![2, 2, 3, 2]);
            assert_eq!(ans, d1 + d2);
        }

        #[test]
        fn test_digits_add_assign() {
            let mut d1 = Digits(vec![0, 0, 1]);
            let d2 = Digits(vec![2, 2, 2, 2]);
            let ans = Digits(vec![2, 2, 3, 2]);
            d1 += d2;
            assert_eq!(ans, d1);
        }
        
        #[test]
        fn test_digits_mul() {
            let d_0 = Digits(vec![0, 0, 1]);
            let d_1 = Digits(vec![0, 0, 0, 1]);
            let ans = Digits(vec![0, 0, 0, 0, 0, 1]);
            assert_eq!(ans, d_0 * d_1)
        }
        
        #[test]
        fn test_digits_from_u32() {
            let v: u32 = 321;
            let ans = Digits(vec![1, 2, 3]);
            assert_eq!(ans, v.into());
        }

        #[test]
        fn test_digits_from_pow2() {
            let p: usize = 32;
            let ans = Digits(vec![6, 9, 2, 7, 6, 9, 4, 9, 2, 4]);  // 4,294,967,295
            assert_eq!(ans, Digits::from_pow_2(p));
        }
    }
}

