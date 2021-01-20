"""

     QTLplot


A module for generating PyPlot-based 1D-, 2D-plots for LOD scores (or effects).

"""
module QTLplot

using PyPlot
import Statistics:median


"""
  
    layers(chr::Array{Any,1},pos::Array{Float64,1},lod::Array{Float64,2})

Creates a struct of arrays for plotting LOD scores, (or main, interaction) effects

# Argument

- `chr` : A vector of strings or numbers indicating Chromosome names, ex. 1,2,3,... or 1K,1N,2K,2N,...
- `pos` : A vector of marker positions.
- `lod` : A matrix of LOD scores obtained from 1d- or 2d-genome scan.  Can be a matrix of effects (main, or interaction).

"""
struct layers
    chr::Array{Any,1}
    pos::Array{Float64,1}
    lod::Array{Float64,2}
end


"""
    
      plot1d(xx::layers;title= " ",ylabel="LOD",yint=[],yint_color=["red"],Legend=[],fontsize="medium",loc="upper right") 


Generates one or more graphs of LOD scores (or effects) obtained from 1d-genome scan on a single plot.

# Arguments

- `xx` : A type of [`layers`](@ref) that consists of chromosome names, marker positions, and a matrix of LODs(or effects).

## Keyword arguements,

- `title` : A string of title. Default is blank.
- `yint` :  A vector of y-intercept(s).
- `yint_color` : A vector of colors (strings) of y-intercepts in yint.
- `Legend` : A vector of titles corresponding to graphs in `layers`.
- `fontsize` : A string or number to set `Legend` fontsize. i.e. "small", "medium", "large", or any integer, '15'.
- `loc` : A string of Legend's position. Default is "upper right".


"""
function plot1d(xx::layers;title= " ",ylabel="LOD",yint=[],yint_color=["red"],Legend=[],fontsize="medium",loc="upper right")

Chr=unique(xx.chr); nchr=length(Chr);np=size(xx.lod,2)
 #generating a set of line segments

idx=findall(xx.chr.==Chr[1])
line_seg=Any[collect(zip(idx,xx.lod[idx,j])) for j=1:size(xx.lod,2)]
#generating major ticks (Mticks), minor ticks(mticks)
    Mticks=[0;idx[end]]; mticks=[median(idx)]

for i=2:nchr
         idx=findall(xx.chr.==Chr[i])
        append!(line_seg,[collect(zip(idx,xx.lod[idx,j])) for j=1:size(xx.lod,2)])
         Mticks=[Mticks; idx[end]]
         mticks=[mticks; median(idx)]
end

#Mticks=collect(Iterators.take(Mticks,nchr-1))
############
##  Plot  ##
############
fig=figure(figsize=[20,10])
ax=subplot(1,1,1) # creates a subplot with just one graphic
    #major labels
Mnames=["" for j=1:nchr]; 
ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(Mticks)) # Set interval of major ticks
ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(Mnames))

ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(mticks)) # Set interval of major ticks
ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(Chr))
    
## colors
#c = Vector{Int}[[0,1,0],[0,1,0],[0,1,0],[1, 1, 0]]
c= repeat([rand(3) for j=1:np],nchr)
# Assemble everything into a LineCollection
line_segments = matplotlib.collections.LineCollection(line_seg, colors=c)

# mx = matplotlib.ticker.MultipleLocator(1) # Define interval of minor ticks
# ax.xaxis.set_minor_locator(mx) # Set interval of minor ticks

My = matplotlib.ticker.MultipleLocator(5) # Define interval of major ticks
ax.yaxis.set_major_locator(My) # Set interval of major ticks

my = matplotlib.ticker.MultipleLocator(1) # Define interval of minor ticks
ax.yaxis.set_minor_locator(my) # Set interval of minor ticks

#horizontal line
#yint=[6.3869; 7.7035]
   if (length(yint)!=0)
      if(length(yint_color)>1)
         for j=1:length(yint)
          ax.axhline(y=yint[j],linewidth=0.8,linestyle="dashed",color=yint_color[j])
         end
       else
         for j=1:length(yint)
        ax.axhline(y=yint[j],linewidth=0.8,linestyle="dashed",color=yint_color[1])
         end   
       end
  end
# #legend
# #lbox= Array{PyCall.PyObject,1}(undef,np);
    if (length(Legend)!=0)
     lbox=[];
 #plegend=["Scan 1"]
       for l=1:np
        lbox0= matplotlib.lines.Line2D([], [], color=c[l],label=Legend[l])  
        lbox=vcat(lbox,lbox0)
       end
     legend(handles=lbox, loc=loc,fontsize=fontsize)
    end
#legend(handles=lbox, bbox_to_anchor=(1, 1))
#fig, ax = subplots(figsize=(10,10))
ax.add_collection(line_segments)
ax.axis("image")
ax.axis("tight")
ax.set(xlabel="Chromosome", ylabel=ylabel, title=title)
ax.grid("on")
# gcf();

end


## 2d-plot
##plotting 2D-scan PyPlot
"""

        plot2d(S::layers)

Generates 2-d heatmap plots of LOD scores from 2d-genome scan
 
# Argument

- `S`: a type of 'layers' that consists of chromosome names, marker positions, and a matrix of LODs

"""
function plot2d(S::layers)
Chr=unique(S.chr);
for i=1:length(Chr)
    cidx=findall(S.chr.==Chr[i])
    Chrom=Array(Symmetric(S.lod[cidx,cidx],:L))
    x,y=S.pos[cidx],S.pos[cidx]
    figure();imshow(Chrom,cmap="jet",interpolation="bicubic",extent=[minimum(x),maximum(x),maximum(y),minimum(y)]
        ,vmin=0.0,vmax=maximum(S.lod));

   eval(Meta.parse(string("""title(string("Chromsome ", """,Chr[i],""")) """)))
        colorbar()
        end
end


#sub_dim is a three digit integer,
#where the first digit is the number of rows, the second the number of columns, and the third the index of the subplot.
"""

     subplot2d(S::layers,sub_dim::Int64;label="Chromosome")


Generates a matrix of 2-d heatmap subplots for LOD scores obtained from 2d-genome scan  

# Arguments 

- `sub_dim`: A two digit integer, where the first digit (m) is the number of rows, the second (n) the number of columns. 
             It returns a m x n matrix of subplots.
- `label`: A string of the title of each subplot. It concatenates each entry of S.chr. 

"""
function subplot2d(S::layers,sub_dim::Int64;label="Chromosome")
Chr=unique(S.chr);
    d=digits(sub_dim,base=10)
    inner_num=d[1]*d[2]
    outer_num=Int(ceil(length(Chr)/inner_num))
    for j=1:outer_num    
      figure();
        for i=1:inner_num
          cidx=findall(S.chr.==Chr[i+inner_num*(j-1)])
          Chrom=Array(Symmetric(S.lod[cidx,cidx],:L))
          x,y=S.pos[cidx],S.pos[cidx]
          position=sub_dim*10+i
        #position=parse(Int,string(nrow,ncol))*10+i
        # position=320+i
          subplot(position)
          imshow(Chrom,cmap="jet",interpolation="bicubic",extent=[minimum(x),maximum(x),maximum(y),minimum(y)],vmin=0.0,vmax=maximum(S.lod));
#          eval(Meta.parse(string("""title(string("Chromsome ", """,Chr[i+inner_num*(j-1)],""")) """)))
           title(string(label*" ",Chr[i+inner_num*(j-1)]))
           colorbar()
        end
    end
end



# for j=1:3
#     figure()
#       for i=1:6
#  Chrom=Array(Symmetric([find(markers.chr.==Chr[i+6*(j-1)]),find(markers.chr.==Chr[i+6*(j-1)])],:L))
#         position=230+i
#         subplot(position)
#     imshow(Chrom,cmap="jet",interpolation="bicubic",vmin=0.0,vmax=maximum(lod2_h));
#    eval(parse(string(""" title("HEIGHT """, Chr[i+6*(j-1)],""" "); colorbar()""")))   
#         end
# end


# export layers, plot1d, plot2d, subplot2d
end