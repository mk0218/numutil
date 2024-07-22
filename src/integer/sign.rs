use std::{
    cmp::Ord,
    fmt::Debug,
    ops::{Mul, MulAssign, Not}
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sign {
    PLUS,
    MINUS,
}

impl Not for Sign {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            Sign::PLUS => Sign::MINUS,
            Sign::MINUS => Sign::PLUS,
        }
    }
}

impl Mul for Sign {
    type Output = Self;
    
    fn mul(self, rhs: Self) -> Self::Output {
        if self == rhs {
            Sign::PLUS
        } else {
            Sign::MINUS
        }
    }
}

impl MulAssign for Sign {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Sign {
    pub fn of<T>(v: T) -> Self
    where
        T: Ord + TryFrom<u8>,
        <T as TryFrom<u8>>::Error: Debug,
    {
        let zero: T = 0_u8.try_into().unwrap();
        if v >= zero {
            Sign::PLUS
        } else {
            Sign::MINUS
        }
    }
}
