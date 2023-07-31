function D = imdog(I, hpSpread, lpSpread)

D = imgaussfilt(I, hpSpread) - imgaussfilt(I, lpSpread);