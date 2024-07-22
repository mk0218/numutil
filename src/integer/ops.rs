use std::{
    cmp::{min, max},
    ops
};

use super::*;

impl Integer {
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

    // fn mul_abs(self: &Self, other: &Self) -> Self {

    // }
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

impl ops::Mul for Integer {
    type Output = Integer;

    fn mul(self, rhs: Self) -> Self::Output {
        0_u8.into()
    }
}

#[cfg(test)]
mod test_utils {
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
mod test_add {
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
mod test_sub {
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

// #[cfg(test)]
// mod test_mul {
//     fn 
// }
