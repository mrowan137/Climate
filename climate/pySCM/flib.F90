subroutine co2ems(oceanresponse, bioresponse, n, & !input
                  emsco2,l, & !input
                  atmosco2) !output
    
  real(8), dimension(l) :: emsco2
  real(8), dimension(n) :: oceanresponse, bioresponse
  integer n, l
  
  real(8), dimension(l) :: atmosco2
  
  !f2py intent(in) :: oceanresponse, bioresponse, emsco2
  !f2py intent(hide), depend(emsco2) :: l = shape(emsco2)
  !f2py intent(hide), depend(bioresponse) :: n = shape(bioresponse)
  !f2py intent(out) atmosco2

  integer i,j
  real(8) :: pgcperppm, tc, a1, a2, a3, a4, a5,&
       delta, xatmosbio, hold,&
       airseagasexchangecoeff,biospherenpp0,co2fertfactor,co2ppm0
  real(8), dimension(l) :: seawaterpco2, surfaceoceandic, atmosbioflux, atmosseaflux
  !define constants
  pgcperppm=2.123_8
  tc=18.1716_8                       
  a1=(1.5568_8-1.3993D-2*tc)
  a2=(7.4706_8-0.20207_8*tc)*1D-3
  a3=-(1.2748_8-0.12015_8*tc)*1D-5
  a4=(2.4491_8-0.12639_8*tc)*1D-7
  a5=-(1.5468_8-0.15326_8*tc)*1D-10
  airseagasexchangecoeff=0.1042_8
  biospherenpp0=60.0_8
  co2fertfactor=0.287_8
  co2ppm0=278.305_8
  
  !initialize arrays and vars
  seawaterpco2(:)=0._8
  surfaceoceandic(:)=0._8
  atmosbioflux(:)=0._8
  atmosseaflux(:)=0._8

  delta=0._8
  xatmosbio=0._8
  hold=0._8
  
  ! Loop through columns and rows and threshold the image.
  do i=1,l-1
     if(i>0) then
        seawaterpco2(i)=surfaceoceandic(i) &
             *(a1+surfaceoceandic(i) &
             *(a2+surfaceoceandic(i) &
             *(a3+surfaceoceandic(i) &
             *(a4+surfaceoceandic(i) &
             *a5))))
     endif
     atmosseaflux(i) = airseagasexchangecoeff*(atmosco2(i)-seawaterpco2(i))
     delta=biospherenpp0*co2fertfactor*log(1._8+(atmosco2(i)/co2ppm0))/pgcperppm-xatmosbio
     xatmosbio = xatmosbio + delta
     atmosbioflux(i) = atmosbioflux(i) + xatmosbio
     do j=(i+1),l
        hold = surfaceoceandic(j)
        hold = hold + atmosseaflux(i) * oceanresponse(j-i+1)
        surfaceoceandic(j) = hold
     enddo
     do j=(i+1),l
        atmosbioflux(j) = atmosbioflux(j) - xatmosbio * bioresponse(j-i+1)
     enddo
     
     atmosco2(i+1)=atmosco2(i)+(emsco2(i)/pgcperppm)-atmosseaflux(i)-atmosbioflux(i)
     
  enddo
  
end subroutine co2ems 

subroutine calctchange(tempresfunc, n, &
                       radforcing, l,&
                       res)

  real(8), dimension(n) :: tempresfunc
  real(8), dimension(l) :: radforcing
  integer n,l
  real(8), dimension(l) :: res

  !f2py intent(in) :: tempresfunc, radforcing
  !f2py intent(hide), depend(radforcing) :: l-shape(radforcing)
  !f2py intent(hide), depend(tempresfunc) :: n=shape(tempresfunc)
  !f2py intent(out) res

  integer i,j
  real(8) :: climatesensitivity

  !initialize
  res(:)=0._8
  climatesensitivity=1.1_8

  do i=1,l
     do j=i,l
        res(j) = res(j) + radforcing(i) * tempresfunc(j-i+1)
     enddo
  enddo
  res = res*climatesensitivity
end subroutine calctchange
