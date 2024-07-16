use std::fmt;

#[derive(Clone, Copy, Debug)]
enum Sign {
    PLUS,
    MINUS,
}

use digits::Digits;

struct Integer {
    sign: Sign,
    value: Vec<u32>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer_zero_fmt() {
        assert_eq!("0", &format!("{}", Integer { sign: Sign::PLUS, value: vec![]}));
        assert_eq!("0", &format!("{}", Integer { sign: Sign::MINUS, value: vec![]}))
    }

    #[test]
    fn test_integer_plus_fmt() {
        assert_eq!("123", &format!("{}", Integer { sign: Sign::PLUS, value: vec![123]}));
    }

    #[test]
    fn test_integer_minus_fmt() {
        assert_eq!("-12312", &format!("-{}", Integer { sign: Sign::PLUS, value: vec![12312]}));
    }

    #[test]
    fn test_integer_large() {
        assert_eq!("4294967296", format!("{}", Integer { sign: Sign::PLUS, value: vec![0, 1] }))
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

