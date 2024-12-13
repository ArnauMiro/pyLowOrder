#!/usr/bin/env python
#
# Testsuite - What about dragons?!
#
# 20/09/2024
from __future__ import print_function

import random


def dragon(flush=True):
	'''
	https://www.asciiart.eu/mythology/dragons
	'''
	print('''
                _ ___                /^^\ /^\  /^^\_
    _          _@)@) \            ,,/ '` ~ `'~~ ', `\.
  _/o\_ _ _ _/~`.`...'~\        ./~~..,'`','',.,' '  ~:
 / `,'.~,~.~  .   , . , ~|,   ,/ .,' , ,. .. ,,.   `,  ~\_
( ' _' _ '_` _  '  .    , `\_/ .' ..' '  `  `   `..  `,   \_
 ~V~ V~ V~ V~ ~\ `   ' .  '    , ' .,.,''`.,.''`.,.``. ',   \_
  _/\ /\ /\ /\_/, . ' ,   `_/~\_ .' .,. ,, , _/~\_ `. `. '.,  \_
 < ~ ~ '~`'~'`, .,  .   `_: ::: \_ '      `_/ ::: \_ `.,' . ',  \_
  \ ' `_  '`_    _    ',/ _::_::_ \ _    _/ _::_::_ \   `.,'.,`., \-,-,-,_,_,
   `'~~ `'~~ `'~~ `'~~  \(_)(_)(_)/  `~~' \(_)(_)(_)/ ~'`\_.._,._,'_;_;_;_;_;

	''',flush=flush)
	print('Remember: The dragon almost never lies ',flush=flush)
	if random.randint(1, 100) > 90:
		print('SINNER: Are you ACTUALLY reading your tests output ?? !! ',flush=flush)
	if random.randint(1, 100) > 93: # randomly print some stuff to keep audience interested
		if random.randint(1,100) > 50:
			print('  Remember: Superman flies but does not exist !',flush=flush)
		else:
			print('  Remember: Tests are the voice of your conscience !',flush=flush)


def dragonAlmostOK(flush=True):
	'''
	https://www.asciiart.eu/mythology/dragons
	'''
	print('''
                ^    ^
               / \  //\ 
    |\___/|      /   \//  .\ 
    /O  O  \__  /    //  | \ \ 
    /     /  \/_/    //   |  \  \ 
    @___@'    \/_   //    |   \   \ 
    |       \/_ //     |    \    \ 
    |        \///      |     \     \ 
    _|_ /   )  //       |      \     _\ 
    '/,_ _ _/  ( ; -.    |    _ _\.-~        .-~~~^-.
    ,-{        _      `-.|.-~-.           .~         `.
    '/\      /                 ~-. _ .-~      .-~^-.  \ 
     `.   {            }                   /      \  \ 
    .----~-.\        \-'                 .~         \  `. \^-.
    ///.----..>    c   \             _ -~             `.  ^-`   ^-_
    ///-._ _ _ _ _ _ _}^ - - - - ~                     ~--,   .-~
            Some of the tests are OK...                   /.-'
	''',flush=flush)  


def dragonAllOK(flush=True):
	print('''
    (  )   /\   _                 (
    \ |  (  \ ( \.(               )                      _____
    \  \ \  `  `   ) \             (  ___                 / _   \\
    (_`    \+   . x  ( .\            \/   \____-----------/ (o)   \_
    - .-               \+  ;          (  O                           \____
    		  )        \_____________  `              \  /
    (__                +- .( -'.- <. - _  VVVVVVV VV V\                 \/
    (_____            ._._: <_ - <- _  (--  _AAAAAAA__A_/                  |
    .    /./.+-  . .- /  +--  - .     \______________//_              \_______
    (__ ' /x  / x _/ (                                  \___'          \     /
    , x / ( '  . / .  /                                      |           \   /
    /  /  _/ /    +                                      /              \/
    '  (__/                   ohlala !                   /                  \ 
    
	''',flush=flush)


def dragonAngry(flush=True):
	print('''
                              ==(W{==========-      /===-                        
            SINNER!!          ||  (.--.)         /===-_---~~~~~~~~~------____  
                              | \_,|**|,__      |===-~___                _,-' `
                 -==\\        `\ ' `--'   ),    `//~\\   ~~~~`---.___.-~~      
                   -==|        /`\_. .__/\ \    | |  \\           _-~`         
       __--~~~  ,-/-==\\      (   | .  |~~~~|   | |   `\        ,'             
    _-~       /'    |  \\     )__/==0==-\<>/   / /      \      /               
  .'        /       |   \\      /~\___/~~\/  /' /        \   /'                
 /  ____  /         |    \`\.__/-~~   \  |_/'  /          \/'                  
/-'~    ~~~~~---__  |     ~-/~         ( )   /'        _--~`                   
                  \_|      /        _) | ;  ),   __--~~                        
                	'~~--_/      _-~/- |/ \   '-~ \                            
                   {\__--_/}    / \\_>-|)<__\      \                           
                   /'   (_/  _-~  | |__>--<__|      |                          
                  |   _/) )-~     | |__>--<__|      |                          
                  / /~ ,_/       / /__>---<__/      |                          
                 o-o _//        /-~_>---<__-~      /                           
                 (^(~          /~_>---<__-      _-~                            
                ,/|           /__>--<__/     _-~                               
             ,//('(          |__>--<__|     /                  .----_          
            ( ( '))          |__>--<__|    |                 /' _---_~\        
         `-)) )) (           |__>--<__|    |               /'  /     ~\`\      
        ,/,'//( (             \__>--<__\    \            /'  //        ||      
      ,( ( ((, ))              ~-__>--<_~-_  ~--____---~' _/'/        /'       
    `~/  )` ) ,/|                 ~-_~>--<_/-__       __-~ _/                  
  ._-~//( )/ )) `                    ~~-'_/_/ /~~~~~~~__--~                    
   ;'( ')/ ,)(                              ~~~~~~~~~~                         
  ' ') '( (/                                                                   
    '   '  `
    ERRORS!!!!!!!!!!!!!!!!
	''',flush=flush)
	print('Check your code before pushing or I will hunt you down!',flush=flush)