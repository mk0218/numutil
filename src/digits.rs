use std::cmp::max;
use std::ops;

#[derive(PartialEq, Clone, Debug)]
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

impl From<u8> for Digits {
    fn from(value: u8) -> Self {
        let mut v = value;
        let mut d: Vec<u8> = vec![];

        while v > 0 {
            d.push((v % 10).try_into().unwrap());
            v /= 10;
        }

        Digits(d.into_iter().rev().collect())
    }
}

impl From<&u8> for Digits {
    fn from(value: &u8) -> Self {
        let mut v = *value;
        let mut d: Vec<u8> = vec![];

        while v > 0 {
            d.push((v % 10).try_into().unwrap());
            v /= 10;
        }

        Digits(d.into_iter().rev().collect())
    }
}

impl From<&Vec<u8>> for Digits {
    fn from(value: &Vec<u8>) -> Self {
        dbg!(value);
        let mut digits: Digits = 0_u8.into();
        
        let base = Digits(vec![2, 5, 6]);
        let mut factor = Digits(vec![1]);
        
        for (i, v) in value.iter().enumerate() {
            let d: Digits = v.into();
            digits += d * factor.clone();
            factor = factor * base.clone();
            dbg!(&digits);
        }
        digits
    }
}

impl Digits {
    pub fn pow_10(p: usize) -> Self {
        let mut v: Vec<u8> = vec![0; p + 1];
        v[0] = 1;
        Digits(v)
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
    fn test_digits_pow_10_zero() {
        let ans = Digits(vec![1]);
        let res = Digits::pow_10(0);
        assert_eq!(ans, res);
    }
}