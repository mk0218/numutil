mod cmp;
mod convert;
mod ops;
mod sign;

use std::fmt::{self, Debug};

use crate::digits::Digits;
use sign::Sign;

#[derive(Debug, Clone, Eq)]
pub struct Integer {
    sign: Sign,
    value: Vec<u8>,
}

impl Integer {    
    pub fn abs(self: &Self) -> Self {
        let Integer { value, .. } = &self;

        Integer {
            sign: Sign::PLUS,
            value: value.to_vec()
        }
    }
    
    fn with_sign(self, sign: Sign) -> Self {
        Integer { sign, value: self.value }
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
