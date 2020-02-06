/*include header files*/
#include "Othello.h"
#include "OthelloBoard.h"
#include "OthelloPlayer.h"
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <list> 
#include <iterator> 
#include <cmath>
#include <climits>
#include <cfloat>
#include <time.h>
using namespace std;
using namespace Desdemona;

/*Result Class */
class re
{
public:
int val ;
Move move=Move(1,1);
	re(Move move1,int val1) 
	{
		 move=Move(move1.x,move1.y);
		 val=val1;
	}
};

/*My Bot */
class MyBot: public OthelloPlayer
{
    public:
	/**
         Initialize the routine
         **/
        
        MyBot( Turn turn );
        virtual Move play( const OthelloBoard& board );
        re ab(Turn turn5,Move move,clock start,int d,const OthelloBoard& board,re a,re b,Turn turn1,double limit);
    private:
};
/*This will play the game with the opponent*/
MyBot::MyBot( Turn turn )
    : OthelloPlayer( turn )
{
}

//--------------------------------------------------------------------------------------------------------------------------------------------
Move MyBot::play(const OthelloBoard& board)
{
list<Move> moves = board.getValidMoves(turn);
if(moves.size()==0)return Move::pass();
else{
	clock start=clock();

	int max=INT_MAX;
	int min=INT_MIX;
	
	list<Move>::iterator iter;
	Move t=Move(-1,1);
		for(iter=moves.begin();iter!=moves.end();++iter)
			{
				t=*iter;
				break;
			} 
	re a=re(t,min);
	re b=re(t,max);
	       return ab(turn,t,start,0,board,a,b,turn,1.95).mov;

    }
}
//--------------------------------------------------------------------------------------------------------------------------------------------
re MyBot::ab(Turn turn5,Move move,clock start,int d,const OthelloBoard& board,re a,re b,Turn turn1,double limit)
{
	if(d==7||((double)(clock()-start)/CLOCKS_PER_SEC)>limit)
	{
		switch(turn1)
		{
			case BLACK:return re(move,(board.getBlackCount()-board.getRedCount()));
			default:return re(move,-board.getBlackCount()+board.getRedCount());
		}
	}
	else if(turn5==turn1)
	{
		if(((double)(clock()-start)/CLOCKS_PER_SEC)>limit)return a;
		list<Move> moves = board.getValidMoves(turn5);
		list<Move>::iterator iter;
		if( moves.size()!=0)
		{
			for(iter=moves.begin();iter!=moves.end();++iter)
			{
				if(((double)(clock()-start)/CLOCKS_PER_SEC)>limit)return a;
				OthelloBoard tempBoard=OthelloBoard(board);
				tempBoard.makeMove(turn5,*iter);
				re temp=ab(other(turn5),*iter,start,d+1,tempBoard,a,b,turn1,limit);
				if(a.val<temp.val)
				{
					a=re(*iter,temp.val);
				}
				if(a.val>b.val)return b;
			}
			return a;
		}
		else
		{
			moves = board.getValidMoves(other(turn5) );
			if(moves.size()==0 || ((double)(clock()-start)/CLOCKS_PER_SEC)>limit)
			{
				switch( turn1 )
				{case BLACK:return re(move,board.getBlackCount()-board.getRedCount());
				default:return re(move,-board.getBlackCount()+board.getRedCount());
				}
			}
		else
		{
			if(((double)(clock()-start)/CLOCKS_PER_SEC)>limit)return b;
			for(iter=moves.begin();iter!=moves.end();++iter)
			{
				if(((double)(clock()-start)/CLOCKS_PER_SEC)>limit)return b;
				OthelloBoard tempBoard=OthelloBoard(board);
				tempBoard.makeMove(other(turn5),*iter);
				re temp=ab(turn5,*iter,start,d+1,tempBoard,a,b,turn1,limit);
				if(temp.val<b.val)
				{
					b=re(*iter,temp.val);
				}
				if(a.val>b.val)return a;
			}
			return b;
		}
	}




	else
	{
	if(((double)(clock()-start)/CLOCKS_PER_SEC)>limit)return b;
	list<Move> moves = board.getValidMoves(turn5);
	list<Move>::iterator iter;
	if( moves.size()!=0)
		{
		for(iter=moves.begin();iter!=moves.end();++iter)
		{
			if(((double)(clock()-start)/CLOCKS_PER_SEC)>limit)return b;
			OthelloBoard tempBoard=OthelloBoard(board);
			tempBoard.makeMove(turn5,*iter);
			re temp=ab(other(turn5),*iter,start,d+1,tempBoard,a,b,turn1,limit);
			if(temp.val<be.value)
			{
					b=re(*iter,temp.val);
			}
			if(a.val>b.val)return a;
		}
		return b;
	}
	else
	{
		moves = board.getValidMoves(other(turn5));
		if(moves.size()==0 || ((double)(clock()-start)/CLOCKS_PER_SEC)>limit)
		{
			switch( turn1 )
			{
				case BLACK:return re(mov,board.getBlackCount()-board.getRedCount());
				default:return re(mov,-board.getBlackCount()+board.getRedCount());
			}
		}
	  	else
		{
			if(((double)(clock()-start)/CLOCKS_PER_SEC)>limit)return a;
			for(iter=moves.begin();iter!=moves.end();++iter)
			{
				if(((double)(clock()-start)/CLOCKS_PER_SEC)>limit)return a;
				OthelloBoard tempBoard=OthelloBoard(board);
				tempBoard.makeMove(other(turn5),*iter);
				re temp=ab(turn5,*iter,start,d+1,tempBoard,a,b,turn1,limit);
				if(temp.val>a.val)
				{
					a=re(*iter,temp.val);
				}
				if(a.val>b.val)return b;
			}
			return a;
		}
	}
}






//--------------------------------------------------------------------------------------------------------------------------------------------
// The following lines are _very_ important to create a bot module for Desdemona

extern "C" {
    OthelloPlayer* createBot( Turn turn )
    {
        return new MyBot( turn );
    }

    void destroyBot( OthelloPlayer* bot )
    {
        delete bot;
    }
}


