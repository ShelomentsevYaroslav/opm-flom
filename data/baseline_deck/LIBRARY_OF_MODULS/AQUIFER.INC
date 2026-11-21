AQUFETP
-- No. DatumDepth Pinit  WaterVolume  Ctotal     PI      PVTTableNo  SaltConc
--     m         Bar     Sm3        1/bar     sm3/D/bar              
    1  2490.00    1*    1.6E+9     7.25E-05   2320.000        1        .00000 /
-- Pinit defaulted -> in equilibrium with the reservoir
-- Connection of the aquifer with the reservoir
AQUANCON
-- Aqu.No I1-I2  J1-J2  K1-K2 ResFace  AquInfluxCoeff. CoeffMultiplier
    1      1  1   1 15   1  5  'I-'        1*              1.0   /
    1      1 15  15 15   1  5  'J+'        1*              1.0   /
    1      1 15   1 15   6  6  'Z-'        1*              1.0   /
/