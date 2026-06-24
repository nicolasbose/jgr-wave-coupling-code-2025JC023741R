/*
** svn $Id: ceara_test.h 25 2007-04-09 23:43:58Z jcwarner $
*******************************************************************************
** Copyright (c) 2002-2007 The ROMS/TOMS Group                               **
**   Licensed under a MIT/X style license                                    **
**   See License_ROMS.txt                                                    **
*******************************************************************************
**
** Options for ATMOSPHERE-WAVE COUPLING PORTO ILHA
**
** Application flag:   ATMOSPHERE-WAVE COUPLING PORTO ILHA 
*/


#define NESTING
#define WRF_MODEL
#define SWAN_MODEL
#define MCT_LIB
#define MCT_INTERP_WV2AT

#if defined WRF_MODEL && (defined SWAN_MODEL || defined WW3_MODEL)
# define DRAGLIM_DAVIS
# define DRENNAN
#endif

