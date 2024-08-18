use std::{
    cmp::{max, min},
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

    /// Performs u8 integer addition.
    /// Returns (carry, result mod 2^8).
    fn add_u8(v1: u8, v2: u8) -> (u8, u8) {
        let half = 1 << 7;
        let (h1, v1) = (v1 / half, v1 % half);
        let (h2, v2) = (v2 / half, v2 % half);
        let (h3, v) = ((v1 + v2) / half, (v1 + v2) % half);

        match h1 + h2 + h3 {
            3.. => (1, v + half),
            2 => (1, v),
            1 => (0, v + half),
            0 => (0, v),
        }
    }

    /// Returns `v * ((2 ^ 8) ^ cnt)`.
    fn shl_value(v: Integer, cnt: usize) -> Integer {
        let Integer { sign, mut value } = v;

        if cnt > 0 {
            let mut result = vec![0; cnt];
            result.append(&mut value);
            value = result;
        }

        Integer { sign, value }
    }

    /// Returns `v / ((2 ^ 8) ^ cnt)`.
    fn shr_value(v: Integer, cnt: usize) -> Integer {
        let Integer { sign, mut value } = v;

        if cnt > value.len() {
            value.clear();
        } else if cnt > 0 {
            value = value[cnt..].to_vec();
        }

        Integer { sign, value }
    }

    fn share_rest(self: Integer, rhs: Integer) -> (Integer, Integer) {
        fn _abs(v1: Integer, v2: Integer) -> (Integer, Integer) {
            if v1 < v2 {
                (0.into(), v1)
            }  else if v1 == v2 {
                (1.into(), 0.into())
            }  else {
                let shr = |v, n| Integer::shr_value(v, n);
                let shl = |v, n| Integer::shl_value(v, n);

                let v1_shr = shr(v1.clone(), 1);

                let (share_prev, rest_prev) = _abs(v1_shr.clone(), v2.clone());

                let mut rest = shl(rest_prev, 1) + (v1 - shl(v1_shr, 1));
                let mut cnt = 0;

                while rest >= v2 {
                    rest = rest - v2.clone();
                    cnt += 1;
                }

                let share = shl(share_prev, 1) + cnt.into();

                (share, rest)
            }
        }

        let (share, rest) = _abs(self.abs(), rhs.abs());

        let (sign_share, sign_rest) = match (self.sign, rhs.sign) {
            (Sign::PLUS, Sign::PLUS) => (Sign::PLUS, Sign::PLUS),
            (Sign::PLUS, Sign::MINUS) => (Sign::MINUS, Sign::PLUS),
            (Sign::MINUS, Sign::PLUS) => (Sign::MINUS, Sign::MINUS),
            (Sign::MINUS, Sign::MINUS) => (Sign::PLUS, Sign::MINUS),
        };

        (share.with_sign(sign_share), rest.with_sign(sign_rest))
    }
    
    /// Performs u8 integer multiplication.
    /// Returns (carry, result mod 2^8).
    fn mul_u8(v1: u8, v2: u8) -> (u8, u8) {
        let div_16 = |v| {
            (v >> 4, v - ((v >> 4) << 4))
        };

        dbg!(v1, v2);

        let (v1_l, v1_r) = div_16(v1);
        let (v2_l, v2_r) = div_16(v2);

        let (t1, t4) = (v1_l * v2_l, v1_r * v2_r);
        let (t2, t3) = (v1_l * v2_r, v1_r * v2_l);

        let (t2_l, t2_r) = div_16(t2);
        let (t3_l, t3_r) = div_16(t3);

        let c1 = t1 + t2_l + t3_l;
        let (c2, r1) = Integer::add_u8(t2_r << 4, t3_r << 4);
        let (c3, r2) = Integer::add_u8(r1, t4);

        (c1 + c2 + c3, r2)
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

impl ops::Mul for Integer {
    type Output = Integer;

    fn mul(self, rhs: Self) -> Self::Output {
        let (v1, v2) = (self, rhs);
        let (s1, s2) = (v1.sign, v2.sign);

        if v1 == 0.into() || v2 == 0.into() { 0.into() }
        else if v1.abs() == 1.into() { v2.with_sign(s1 * s2) }
        else if v2.abs() == 1.into() { v1.with_sign(s1 * s2) }
        else if v1.value.len() == 1 && v2.value.len() == 1 {
            let v1: u8 = v1.abs().try_into().unwrap();
            let v2: u8 = v2.abs().try_into().unwrap();

            let value = match Integer::mul_u8(v1, v2) {
                (0, 0) => vec![],
                (0, r) => vec![r],
                (c, r) => vec![r, c],
            };

            Integer { sign: s1 * s2, value }
        } else {
            let (v1, v2) = (v1.value, v2.value);

            let v1_l = Integer { sign: s1, value: v1[..(v1.len() / 2)].to_vec() };
            let v1_r = Integer { sign: s1, value: v1[(v1.len() / 2)..].to_vec() };
            let v2_l = Integer { sign: s2, value: v2[..(v2.len() / 2)].to_vec() };
            let v2_r = Integer { sign: s2, value: v2[(v2.len() / 2)..].to_vec() };

            let (l1, l2) = (v1_l.value.len(), v2_l.value.len());

            let t1 = v1_l.clone() * v2_l.clone();
            let t2 = v1_l.clone() * v2_r.clone();
            let t3 = v1_r.clone() * v2_l.clone();
            let t4 = v1_r.clone() * v2_r.clone();

            dbg!(&t1, &t2, &t3, &t4);

            t1
                + Integer::shl_value(t2, l2)
                + Integer::shl_value(t3, l1)
                + Integer::shl_value(t4, l1 + l2)
        }
    }
}

impl ops::Div for Integer {
    type Output = Integer;

    fn div(self, rhs: Self) -> Self::Output {
        Integer::share_rest(self, rhs).0
    }
}

impl ops::Rem for Integer {
    type Output = Integer;

    fn rem(self, rhs: Self) -> Self::Output {
        Integer::share_rest(self, rhs).1
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

    #[test]
    fn add_u8_small_1() {
        let (v1, v2) = (127, 127);
        let res = Integer::add_u8(v1, v2);
        let ans = (0, 254);
        assert_eq!(ans, res);
    }
    
    #[test]
    fn add_u8_large_1() {
        let (v1, v2) = (128, 127);
        let res = Integer::add_u8(v1, v2);
        let ans = (0, 255);
        assert_eq!(ans, res);
    }

    #[test]
    fn add_u8_large_2() {
        let (v1, v2) = (255, 255);
        let res = Integer::add_u8(v1, v2);
        let ans = (1, 254);
        assert_eq!(ans, res);
    }

    #[test]
    fn mul_u8() {
        let (v1, v2) = (15, 15);
        let res = Integer::mul_u8(v1, v2);
        let ans = (0, 225);
        assert_eq!(res, ans);
    }

    #[test]
    fn mul_u8_large2() {
        let (v1, v2) = (200, 15);
        let res = Integer::mul_u8(v1, v2);
        let ans = (11, 184);
        assert_eq!(res, ans);
    }

    #[test]
    fn mul_u8_large3() {
        let (v1, v2) = (255, 255);
        let res = Integer::mul_u8(v1, v2);
        let ans = (254, 1);
        assert_eq!(res, ans);
    }

    #[test]
    fn share_rest() {
        let (v1, v2) = (Integer::from(100), Integer::from(30));
        let res = Integer::share_rest(v1, v2);
        let ans = (3.into(), 10.into());
        assert_eq!(res, ans);
    }

    #[test]
    fn share_rest_pos_neg() {
        let (v1, v2) = (Integer::from(100), Integer::from(-30));
        let res = Integer::share_rest(v1, v2);
        let ans = ((-3).into(), 10.into());
        assert_eq!(res, ans);
    }
    
    #[test]
    fn share_rest_neg_pos() {
        let (v1, v2) = (Integer::from(-100), Integer::from(30));
        let res = Integer::share_rest(v1, v2);
        let ans = ((-3).into(), (-10).into());
        assert_eq!(res, ans);
    }
    
    #[test]
    fn share_rest_neg_neg() {
        let (v1, v2) = (Integer::from(-100), Integer::from(-30));
        let res = Integer::share_rest(v1, v2);
        let ans = (3.into(), (-10).into());
        assert_eq!(res, ans);
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

#[cfg(test)]
mod test_mul {
    use super::*;
    
    #[test]
    fn mul() {
        let (v1, v2) = (Integer::from(100), Integer::from(100));
        let ans = Integer::from(10000);
        assert_eq!(v1 * v2, ans);
    }

    #[test]
    fn mul_large() {
        let (v1, v2) = (Integer::from(10), Integer::from(10_000));
        let ans = Integer::from(100_000);
        assert_eq!(v1 * v2, ans);
    }
    
    #[test]
    fn mul_larger() {
        let (v1, v2) = (Integer::from(10_000), Integer::from(10_000));
        let ans = Integer::from(100_000_000);
        assert_eq!(v1 * v2, ans);
    }

    #[test]
    fn mul_pos_neg() {
        let (v1, v2) = (Integer::from(-10), Integer::from(10_000));
        let ans = Integer::from(-100_000);
        assert_eq!(v1 * v2, ans);
    }

    #[test]
    fn mul_neg_neg() {
        let (v1, v2) = (Integer::from(-10_000), Integer::from(-10_000));
        let ans = Integer::from(100_000_000);
        assert_eq!(v1 * v2, ans);
    }
}
